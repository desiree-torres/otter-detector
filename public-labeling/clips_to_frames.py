"""Cut the motion clips in Supabase into frames that volunteers can label.

Two Supabase projects are involved:
  - clips project   (public-labeling/.env)             -> `clips` bucket + table
  - labeling project (public-labeling/labeling-ui/.env) -> `frames` bucket + `images` table

For every row in `clips` with is_processed = false:
  1. download the mp4 from the private `clips` bucket
  2. extract JPEG frames with ffmpeg (FRAME_FPS, default 1 per second)
  3. upload them to the public `frames` bucket at <clip-name>/<clip-name>_00001.jpg
  4. insert one `images` row per frame (storage_path is relative to the bucket,
     which is what labeling-ui/src/supabase.js frameUrl() expects)
  5. mark the clip is_processed = true

Safe to re-run: uploads upsert, existing images rows are skipped, and finished
clips are not touched again. Needs db/volunteer_schema.sql to have been run.

    ../.venv/bin/python clips_to_frames.py               # all unprocessed clips
    ../.venv/bin/python clips_to_frames.py --limit 2     # try a couple first
    ../.venv/bin/python clips_to_frames.py --fps 0.5     # fewer, less-repetitive frames
    ../.venv/bin/python clips_to_frames.py --watch       # keep going: cut each new clip
                                                         # as soon as the recorder uploads it
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values
from supabase import create_client

HERE = Path(__file__).resolve().parent
CLIPS_ENV = HERE / ".env"
LABEL_ENV = HERE / "labeling-ui" / ".env"

CLIPS_BUCKET = "clips"
FRAMES_BUCKET = "frames"


def client(env_file: Path):
    env = dotenv_values(env_file)
    url, key = env.get("SUPABASE_URL"), env.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit(f"SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in {env_file}")
    return create_client(url, key)


def ensure_frames_bucket(sb):
    if FRAMES_BUCKET not in {b.name for b in sb.storage.list_buckets()}:
        sb.storage.create_bucket(FRAMES_BUCKET, options={"public": True})
        print(f"created public bucket '{FRAMES_BUCKET}'")


def ensure_images_table(sb):
    try:
        sb.table("images").select("id").limit(1).execute()
    except Exception:  # noqa: BLE001
        sys.exit("Table 'images' is missing. Run db/volunteer_schema.sql in the "
                 "Supabase SQL Editor first, then re-run this script.")


def extract(clip_path: Path, out_dir: Path, stem: str, fps: float) -> list[Path]:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(clip_path),
         "-vf", f"fps={fps}", "-q:v", "2", str(out_dir / f"{stem}_%05d.jpg")],
        check=True,
    )
    return sorted(out_dir.glob("*.jpg"))


def process_clip(src, sb, clip: dict, fps: float) -> int:
    stem = Path(clip["filename"]).stem
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mp4 = tmp / clip["filename"]
        mp4.write_bytes(src.storage.from_(CLIPS_BUCKET).download(clip["storage_path"]))
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        frames = extract(mp4, frames_dir, stem, fps)

        paths = [f"{stem}/{f.name}" for f in frames]
        existing = {
            r["storage_path"]
            for r in sb.table("images").select("storage_path").in_("storage_path", paths).execute().data
        }
        new_rows = []
        for f, path in zip(frames, paths):
            sb.storage.from_(FRAMES_BUCKET).upload(
                path, f.read_bytes(),
                {"content-type": "image/jpeg", "upsert": "true"},
            )
            if path not in existing:
                new_rows.append({"filename": f.name, "storage_path": path})
        if new_rows:
            sb.table("images").insert(new_rows).execute()

    src.table("clips").update({"is_processed": True}).eq("id", clip["id"]).execute()
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fps", type=float, default=float(os.getenv("FRAME_FPS", 1)))
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (0 = all)")
    ap.add_argument("--watch", action="store_true",
                    help="keep running and cut new clips as they finish uploading")
    ap.add_argument("--interval", type=float, default=60, help="seconds between checks in --watch mode")
    args = ap.parse_args()

    src = client(CLIPS_ENV)  # where the clips live
    sb = client(LABEL_ENV)   # where volunteers label

    ensure_images_table(sb)
    ensure_frames_bucket(sb)

    if not args.watch:
        run_once(src, sb, args)
        return
    print(f"watching for new clips every {args.interval:.0f}s (Ctrl-C to stop)", flush=True)
    while True:
        try:
            run_once(src, sb, args, quiet=True)
        except Exception as e:  # noqa: BLE001 — a network blip shouldn't stop the watcher
            print(f"check failed ({e}); retrying", flush=True)
        time.sleep(args.interval)


def run_once(src, sb, args, quiet=False):
    q = src.table("clips").select("*").eq("is_processed", False).order("created_at")
    if args.limit:
        q = q.limit(args.limit)
    clips = q.execute().data
    if quiet and not clips:
        return
    print(f"{time.strftime('%H:%M')} {len(clips)} unprocessed clip(s), {args.fps} fps", flush=True)

    total = 0
    for i, clip in enumerate(clips, 1):
        try:
            n = process_clip(src, sb, clip, args.fps)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(clips)}] {clip['filename']}: FAILED ({e})")
            continue
        total += n
        print(f"[{i}/{len(clips)}] {clip['filename']}: {n} frames", flush=True)

    count = sb.table("images").select("id", count="exact").eq("is_labeled", False).limit(1).execute().count
    print(f"done: {total} new frames; {count} images waiting to be labeled", flush=True)


if __name__ == "__main__":
    main()
