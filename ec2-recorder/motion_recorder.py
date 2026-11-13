"""
Motion-triggered live-stream recorder for the otter cam.

Reads a YouTube live stream via ffmpeg (raw frames over a pipe — same pattern as
stream-test.py, which survives YouTube's expiring HLS URLs), watches for motion
with an OpenCV background subtractor, and writes one video clip per motion event
to S3.

State machine:
    IDLE       -> keep a rolling pre-roll buffer, watch for motion
    RECORDING  -> write frames to a clip; keep going until the frame has been
                  quiet for TAIL_SECONDS (or MAX_CLIP_SECONDS is hit)
    (on stop)  -> close the file, upload to S3 in a background thread, delete local

Config is read from environment variables (see the CONFIG block). Nothing is
hardcoded that you'd want to change per-deployment.
"""

import collections
import datetime as dt
import os
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import yt_dlp

# Pick up SUPABASE_* from public-labeling/.env (the repo's one secrets file) when
# running from a checkout. Optional: on EC2, systemd sets env vars directly.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public-labeling", ".env"))
except ImportError:
    pass

# ── Config (all overridable via environment variables) ──────────────────────────
YOUTUBE_URL   = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/watch?v=abbR-Ttd-cA")
# After hours cam https://www.youtube.com/watch?v=sq35qS1iVYU&t=2s
# YOUTUBE_URL   = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/watch?v=abbR-Ttd-cA")
# After hours cam 
# YOUTUBE_URL   = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/watch?v=sq35qS1iVYU&t=2s")
# S3_BUCKET is OPTIONAL. Leave it unset to run in LOCAL mode: clips are written to
# WORK_DIR and kept on disk instead of being uploaded/deleted. Great for testing.
S3_BUCKET     = os.environ.get("S3_BUCKET")
S3_PREFIX     = os.environ.get("S3_PREFIX", "clips")          # key prefix inside the bucket
AWS_REGION    = os.environ.get("AWS_REGION", "us-east-2")

# Path to a Netscape-format cookies.txt exported from a browser logged into YouTube.
# Required on cloud/datacenter IPs (e.g. EC2), which YouTube challenges with a
# "confirm you're not a bot" gate. Unset is fine for local runs from home.
COOKIES_FILE  = os.environ.get("COOKIES_FILE")

# Show a live preview window with the motion mask (local testing only — needs a GUI,
# so keep it off on a headless server). Set PREVIEW=1 to enable.
PREVIEW       = os.environ.get("PREVIEW", "0") == "1"

WIDTH         = int(os.environ.get("WIDTH", 1280))
HEIGHT        = int(os.environ.get("HEIGHT", 720))
FPS           = int(os.environ.get("FPS", 15))               # output/record frame rate
FRAME_SIZE    = WIDTH * HEIGHT * 3                            # bytes per raw bgr24 frame

# Motion sensitivity
MOTION_AREA_FRAC = float(os.environ.get("MOTION_AREA_FRAC", 0.012))  # fraction of frame that must change (raised to ignore water ripples)
VAR_THRESHOLD    = int(os.environ.get("VAR_THRESHOLD", 40))          # MOG2 sensitivity (higher = less sensitive)
WARMUP_FRAMES    = int(os.environ.get("WARMUP_FRAMES", FPS * 5))     # let the background model settle first

# Clip shaping
PREROLL_SECONDS   = float(os.environ.get("PREROLL_SECONDS", 3))      # seconds kept before motion started
TAIL_SECONDS      = float(os.environ.get("TAIL_SECONDS", 15))        # keep recording this long after motion stops
MAX_CLIP_SECONDS  = float(os.environ.get("MAX_CLIP_SECONDS", 600))   # hard cap so a clip can't run forever

HEARTBEAT_SECONDS = float(os.environ.get("HEARTBEAT_SECONDS", 60))    # log "still alive + motion level" while idle

# Clips are named + date-partitioned using this timezone (default: the cam's local
# Pacific time). Handles daylight saving automatically.
LOCAL_TZ = ZoneInfo(os.environ.get("TZ_NAME", "America/Los_Angeles"))

WORK_DIR = os.environ.get("WORK_DIR", "/tmp/otter-clips")
os.makedirs(WORK_DIR, exist_ok=True)

# Clips are encoded by piping raw frames into ffmpeg's libx264 (real H.264) —
# far clearer + smaller than OpenCV's built-in mp4v, and identical on Mac/Linux.
CRF            = int(os.environ.get("CRF", 20))                   # quality: lower = clearer + bigger (18-23 sane)
ENCODER_PRESET = os.environ.get("ENCODER_PRESET", "veryfast")    # CPU vs compression; veryfast suits a t3.small

# Stop on our own once this many seconds of clip footage have been recorded
# (0 = run forever, the EC2 behaviour). E.g. 7200 for "collect ~2 hours of motion".
TARGET_SECONDS = float(os.environ.get("TARGET_SECONDS", 0))

# Only create an S3 client (and require boto3) when we actually have a bucket.
s3 = None
if S3_BUCKET:
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)

# Supabase Storage upload (alternative to S3). Set SUPABASE_URL + SUPABASE_SERVICE_KEY
# (service role — bypasses RLS; never ship it to a browser). Each clip also gets a
# row in the `clips` table (see public-labeling/db/clips_schema.sql) if it exists.
SUPABASE_URL         = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET      = os.environ.get("SUPABASE_BUCKET", "clips")
sb = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    if s3 is not None:
        raise SystemExit("Set S3_BUCKET or SUPABASE_URL, not both.")
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ── Stream plumbing ─────────────────────────────────────────────────────────────
def get_stream_url():
    """Resolve a fresh direct stream URL. YouTube's URLs expire, so we re-resolve
    whenever ffmpeg dies."""
    ydl_opts = {"quiet": True, "format": f"bestvideo[height<={HEIGHT}]/best[height<={HEIGHT}]/best"}
    if COOKIES_FILE:
        ydl_opts["cookiefile"] = COOKIES_FILE
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(YOUTUBE_URL, download=False)
        return info["url"]


def start_ffmpeg(url):
    """Start ffmpeg piping normalized raw BGR frames to stdout."""
    cmd = [
        "ffmpeg",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        # Give up if the stream sends nothing for 30 s. Without this a stalled
        # connection blocks the read forever and the recorder silently hangs;
        # with it ffmpeg exits and the main loop re-resolves the stream.
        "-rw_timeout", "30000000",  # microseconds
        "-i", url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-an",
        "-r", str(FPS),
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)


def read_frame(process):
    """Read exactly one frame worth of bytes off the pipe. Returns an ndarray or
    None if the stream ended / ffmpeg died."""
    raw = process.stdout.read(FRAME_SIZE)
    if len(raw) < FRAME_SIZE:
        return None
    return np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))


def start_encoder(path):
    """Start an ffmpeg process that encodes raw BGR frames (written to its stdin)
    into an H.264 mp4. Replaces OpenCV's mp4v writer for much clearer output."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", ENCODER_PRESET, "-crf", str(CRF),
        "-pix_fmt", "yuv420p",   # required for the clip to play in browsers/QuickTime
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ── S3 upload ────────────────────────────────────────────────────────────────────
def clip_key(fname):
    """Date-partition from the filename (otters-YYYY-MM-DD-...) so key + name agree."""
    try:
        _, yyyy, mm, dd, *_ = fname.split("-")
        return f"{S3_PREFIX}/{yyyy}/{mm}/{dd}/{fname}"
    except ValueError:
        return f"{S3_PREFIX}/{dt.datetime.now(LOCAL_TZ):%Y/%m/%d}/{fname}"


def upload_to_supabase(local_path, duration_s):
    """Upload one clip to Supabase Storage + record it in the `clips` table.
    Deletes the local file only on a successful upload, so nothing is lost."""
    fname = os.path.basename(local_path)
    key = clip_key(fname).split("/", 1)[1]   # bucket is already "clips" — drop the S3 prefix
    size = os.path.getsize(local_path)
    for attempt in range(3):
        try:
            with open(local_path, "rb") as f:
                sb.storage.from_(SUPABASE_BUCKET).upload(
                    key, f, {"content-type": "video/mp4", "upsert": "true"}
                )
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"[sb] upload FAILED for {fname} (kept locally): {e}", flush=True)
                return
            time.sleep(5)
    print(f"[sb] uploaded {SUPABASE_BUCKET}/{key} ({size / 1e6:.1f} MB, {duration_s:.0f}s)", flush=True)
    try:
        sb.table("clips").upsert({
            "filename": fname,
            "storage_path": key,
            "duration_s": round(duration_s, 1),
            "size_bytes": size,
        }, on_conflict="storage_path").execute()
    except Exception as e:  # noqa: BLE001 — table is optional; storage is the source of truth
        print(f"[sb] clips row not written ({e})", flush=True)
    try:
        os.remove(local_path)
    except OSError:
        pass


def upload_and_cleanup(local_path, duration_s=0.0):
    """Upload one finished clip to S3 (date-partitioned key) then delete it locally.
    Runs in a background thread so recording isn't blocked by the upload.

    In LOCAL mode (no S3_BUCKET) this is a no-op — the clip stays on disk in WORK_DIR."""
    if sb is not None:
        upload_to_supabase(local_path, duration_s)
        return
    if s3 is None:
        print(f"[local] kept clip {local_path}", flush=True)
        return
    fname = os.path.basename(local_path)
    key = clip_key(fname)
    try:
        s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
        print(f"[s3] uploaded s3://{S3_BUCKET}/{key}", flush=True)
    except Exception as e:  # noqa: BLE001 — never let an upload error kill the recorder
        print(f"[s3] upload FAILED for {fname}: {e}", flush=True)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    if sb:
        dest = f"supabase {SUPABASE_URL} bucket '{SUPABASE_BUCKET}'"
    elif s3:
        dest = f"s3://{S3_BUCKET}/{S3_PREFIX}/"
    else:
        dest = f"{WORK_DIR}/ (LOCAL mode — no upload)"
    print(f"[boot] recording {YOUTUBE_URL} -> {dest}", flush=True)
    if PREVIEW:
        print("[boot] preview window on — press 'q' in the window to quit", flush=True)

    # detectShadows=False keeps water-ripple shadows from tripping motion; MOG2
    # adapts to the constantly-rippling water and treats it as background over time.
    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=VAR_THRESHOLD, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    min_motion_pixels = MOTION_AREA_FRAC * WIDTH * HEIGHT

    preroll = collections.deque(maxlen=int(PREROLL_SECONDS * FPS))

    process = None
    writer = None
    clip_path = None
    clip_start = 0.0
    last_motion = 0.0
    last_heartbeat = 0.0
    frames_seen = 0
    used_names = set()  # guard against two clips sharing a same-minute filename
    clip_frames = 0
    total_seconds = 0.0   # footage recorded so far (for TARGET_SECONDS)
    uploads = []

    def close_clip():
        nonlocal writer, clip_path, total_seconds
        if writer is not None:
            try:
                writer.stdin.close()   # signal end-of-input; ffmpeg flushes + finalizes the mp4
            except (BrokenPipeError, OSError):
                pass
            writer.wait()              # wait for the file to be fully written before uploading
            duration = clip_frames / FPS
            total_seconds += duration
            progress = f" | total {total_seconds / 60:.1f} min" + (
                f" of {TARGET_SECONDS / 60:g}" if TARGET_SECONDS else "")
            print(f"[clip] closed {os.path.basename(clip_path)} ({duration:.0f}s){progress}", flush=True)
            t = threading.Thread(target=upload_and_cleanup, args=(clip_path, duration), daemon=True)
            t.start()
            uploads.append(t)
            writer = None
            clip_path = None

    def write_frame(f):
        """Send one raw frame to the encoder; swallow the broken pipe if ffmpeg died."""
        nonlocal clip_frames
        try:
            writer.stdin.write(f.tobytes())
            clip_frames += 1
        except (BrokenPipeError, OSError):
            pass

    try:
        while True:
            # (Re)start ffmpeg with a fresh URL whenever it isn't running.
            if process is None or process.poll() is not None:
                close_clip()  # don't leave a half-written clip across a reconnect
                try:
                    url = get_stream_url()
                    process = start_ffmpeg(url)
                    print("[stream] ffmpeg (re)started", flush=True)
                    backsub = cv2.createBackgroundSubtractorMOG2(
                        history=500, varThreshold=VAR_THRESHOLD, detectShadows=False
                    )
                    frames_seen = 0
                except Exception as e:  # noqa: BLE001
                    print(f"[stream] could not start: {e}; retrying in 10s", flush=True)
                    time.sleep(10)
                    continue

            frame = read_frame(process)
            if frame is None:
                print("[stream] frame read failed — restarting ffmpeg", flush=True)
                process.kill()
                process = None
                continue

            frames_seen += 1
            now = time.time()

            # --- motion detection (on a downscaled gray frame for speed) ---
            small = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
            mask = backsub.apply(small)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            motion_pixels = cv2.countNonZero(mask) * 4  # *4 undoes the half-res downscale
            motion = frames_seen > WARMUP_FRAMES and motion_pixels > min_motion_pixels

            # --- heartbeat: prove we're alive while idle (nothing else logs during quiet) ---
            if writer is None and now - last_heartbeat > HEARTBEAT_SECONDS:
                last_heartbeat = now
                state = "warming up" if frames_seen <= WARMUP_FRAMES else "idle, watching"
                print(f"[hb] {state} | motion_px={motion_pixels} (trips at {int(min_motion_pixels)})", flush=True)

            # --- state machine ---
            if writer is None:
                preroll.append(frame)
                if motion:
                    base = f"otters-{dt.datetime.now(LOCAL_TZ):%Y-%m-%d-%H%M}"
                    name, n = base, 2
                    while name in used_names or os.path.exists(os.path.join(WORK_DIR, name + ".mp4")):
                        name, n = f"{base}-{n}", n + 1   # 2nd clip in the same minute -> otters-...-2.mp4
                    used_names.add(name)
                    clip_path = os.path.join(WORK_DIR, name + ".mp4")
                    writer = start_encoder(clip_path)
                    clip_frames = 0
                    for f in preroll:  # flush the pre-roll into the clip
                        write_frame(f)
                    preroll.clear()
                    clip_start = now
                    last_motion = now
                    print(f"[clip] motion -> recording {os.path.basename(clip_path)}", flush=True)
            else:
                write_frame(frame)
                if motion:
                    last_motion = now
                quiet_for = now - last_motion
                too_long = now - clip_start > MAX_CLIP_SECONDS
                if quiet_for > TAIL_SECONDS or too_long:
                    reason = "max length" if too_long else "quiet"
                    print(f"[clip] stopping ({reason})", flush=True)
                    close_clip()
                    if TARGET_SECONDS and total_seconds >= TARGET_SECONDS:
                        print(f"[done] reached target of {TARGET_SECONDS / 60:g} min", flush=True)
                        break

            # --- optional live preview (local testing) ---
            if PREVIEW:
                view = frame.copy()
                recording = writer is not None
                color = (0, 0, 255) if recording else (0, 200, 0)
                label = "REC" if recording else ("MOTION" if motion else "idle")
                cv2.putText(view, f"{label}  motion_px={motion_pixels}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("otter recorder (q=quit)", view)
                cv2.imshow("motion mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    close_clip()
                    break

    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C — finishing current clip", flush=True)
        close_clip()

    if process is not None:
        process.kill()
    if PREVIEW:
        cv2.destroyAllWindows()
    pending = [t for t in uploads if t.is_alive()]
    if pending:
        print(f"[stop] waiting for {len(pending)} upload(s)...", flush=True)
    for t in uploads:
        t.join()
    print(f"[stop] recorded {total_seconds / 60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
