# Video indexing with YOLO11l

Reads FineVideo parquet files and writes one JSON line per second of video
(objects present + people count) using YOLO11l (imgsz 640, conf 0.4).

## System packages (Ubuntu 22.04)
```
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

## Install
```
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (needs internet, run once before the timed run)
```
python prepare.py
```
Downloads `yolo11l.pt` into the repo folder.

## Run
```
python run.py --input <folder of .parquet files> --output results.jsonl
```
Defaults: 6 sampled frames per second of video, batch 32, all visible GPUs.
Progress is kept per parquet file in `results.jsonl.parts/`; delete that folder to start fresh.

## What this repo contains
- `run.py`: the script; `prepare.py`: downloads the weights; `requirements.txt`: pinned packages
- `results.jsonl.gz`: results for all 50 files (1,650 videos, 461,757 seconds, 0 error lines)
- `HARDWARE.md`: the machine the full run used and how long it took
- `WRITEUP.md`: answers to the write-up questions
- `ai_chats/`: AI chat history

## Approach in brief
- Videos are decoded straight from the parquet bytes, with the rotation tag applied.
- About 6 frames per second of video are sampled (not every frame) and run through YOLO11l in batches.
- Decoding runs in a background thread while the GPU does inference, and one worker runs per GPU.
- Results are written per parquet file, so an interrupted run resumes where it stopped.
- Per second: an object counts as present if it is in at least half of the sampled frames, and the people count is the median (rounded down).

## Results
- Full 50 files on a Kaggle notebook (2x T4): about 4.2 h in two runs (see `HARDWARE.md`).
- Correctness check on 8 videos against an every-frame reference: F1 0.985, exact people count 95.7%, within 1 on 99.8%.
- The sampling rate was chosen by comparing 2, 3, 4, 6 and 8 frames per second against that reference.
