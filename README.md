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
