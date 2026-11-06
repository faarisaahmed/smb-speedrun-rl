"""Build the project website (docs/, served by GitHub Pages) from a run's logs and videos.

  python make_site.py            # docs/data.json + docs/media/ from runs/main and videos/

docs/index.html, style.css and app.js are hand-written; this only refreshes
the numbers and copies in the videos (with a poster frame each, so the page
loads fast and fetches a video only when it's played).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess

NES_FPS = 60.0988
CKPT = re.compile(r"\[ckpt\] ([\d,]+) steps, stage (\d), success \{(.*)\}")
POINTS = 400   # training-curve samples kept per series

# The videos on the site: id -> source under videos/ (records and races are added per level).
VIDEOS = {
    "pipe_shortcut": "cool/1-1_found_the_tas_pipe_shortcut_20.00s.mp4",
    "secret_pipe": "cool/1-1_secret_pipe.mp4",
    "progression_1-1": "cool/1-1_record_progression.mp4",
    "progression_1-3": "cool/1-3_record_progression.mp4",
    "first_clear_1-1": "other/1-1_first_clear.mp4",
    "first_clear_1-3": "other/1-3_first_clear.mp4",
    "ghost_1-1_early": "ghosts/1-1_early_training.mp4",
    "ghost_1-1_late": "ghosts/1-1_late_training.mp4",
    "ghost_1-3_jump": "ghosts/1-3_after_learning_the_jump.mp4",
    "ghost_3-2": "ghosts/3-2_300_attempts_at_146M.mp4",
    "ghost_4-3": "ghosts/4-3_300_attempts_at_225M.mp4",
    "ghost_6-2_wall": "ghosts/6-2_the_buzzy_shell_wall_at_430M.mp4",
    "ghost_6-2": "ghosts/6-2_300_attempts_at_890M.mp4",
    "ghost_8-1_early": "ghosts/8-1_300_attempts_at_482M.mp4",
    "ghost_8-1": "ghosts/8-1_300_attempts_at_1256M.mp4",
    "ghost_7-1": "ghosts/7-1_hammer_bros_300_attempts_at_1650M.mp4",
}


def secs(frames):
    return None if frames is None else round(frames / NES_FPS, 2)


def training_curve(log):
    """Per-checkpoint clear rates from train.log, thinned to about POINTS samples."""
    rows = []
    for line in open(log):
        m = CKPT.search(line)
        if m:
            rates = {k: float(v) for k, v in (kv.split(": ") for kv in m.group(3).split(", "))}
            rows.append({"t": int(m.group(1).replace(",", "")), "stage": int(m.group(2)), "rates": rates})
    rows.sort(key=lambda r: r["t"])
    # A resume re-logs steps it already passed; keep the latest line per step.
    rows = list({r["t"]: r for r in rows}.values())
    step = max(1, len(rows) // POINTS)
    return rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else [])


def records(path):
    """Each level's run of record times, from the level start only."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("from_start", True) and not r.get("demo_start"):
            out.setdefault(r["level"], []).append([r["timesteps"], secs(r["frames"])])
    return {lv: sorted(rs) for lv, rs in out.items()}


def races(run_dir, tas):
    """race_tas.py's results against one tracked TAS: per level, both sides' times."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, tas, "*.json"))):
        level = os.path.basename(path)[:-len(".json")]
        target = json.load(open(path))
        race_path = os.path.join(run_dir, f"{tas}_race", f"{level}.json")
        race = json.load(open(race_path)) if os.path.exists(race_path) else None
        out[level] = {
            "tas_flag": secs(target["flag_frame"]), "tas_next": secs(target["next_start"]),
            "agent_flag": secs(race and race["flag_frame"]), "agent_next": secs(race and race["next_start"]),
            "checkpoint": race and int(race["checkpoint"].split("_")[1].split(".")[0]),
        }
    return out


def media(src, dest_dir, name):
    """Copy a video in and grab its poster frame; returns its entry for data.json."""
    dest = os.path.join(dest_dir, name + ".mp4")
    shutil.copyfile(src, dest)
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", src], capture_output=True, text=True).stdout)
    # Ghosts start bunched up at the level start; their poster is from further in.
    at = duration * 0.4 if name.startswith("ghost") else 1.5
    poster = os.path.join(dest_dir, name + ".jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.1f}", "-i", src, "-frames:v", "1",
                    "-q:v", "4", poster], check=True)
    return {"src": f"media/{name}.mp4", "poster": f"media/{name}.jpg", "duration": round(duration, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main")
    p.add_argument("--out", default="docs")
    args = p.parse_args()
    run_dir = os.path.join("runs", args.run)
    status = json.load(open(os.path.join(run_dir, "status.json")))

    media_dir = os.path.join(args.out, "media")
    shutil.rmtree(media_dir, ignore_errors=True)
    os.makedirs(media_dir)
    videos = {key: media(os.path.join("videos", src), media_dir, key)
              for key, src in VIDEOS.items() if os.path.exists(os.path.join("videos", src))}
    for path in glob.glob(os.path.join("videos", "records", "*.mp4")):
        level = os.path.basename(path).split("_")[0]
        videos[f"record_{level}"] = media(path, media_dir, f"record_{level}")
    for path in glob.glob(os.path.join("videos", "races", "*.mp4")):
        name = os.path.basename(path)[:-len(".mp4")]
        videos[f"race_{name}"] = media(path, media_dir, f"race_{name}")

    data = {
        "timesteps": status["timesteps"],
        "levels": status["levels"],
        "holdout": status["holdout"],
        "success": status["success"],
        "best": {lv: secs(f) for lv, f in status["best_frames"].items()},
        "records": records(os.path.join(run_dir, "records.jsonl")),
        "curve": training_curve(os.path.join(run_dir, "train.log")),
        "race": races(run_dir, "tas"),
        "race_small": races(run_dir, "tas_small"),
        "videos": videos,
    }
    json.dump(data, open(os.path.join(args.out, "data.json"), "w"), separators=(",", ":"))
    print(f"wrote {args.out}/data.json and {len(videos)} videos")


if __name__ == "__main__":
    main()
