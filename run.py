import argparse, glob, io, json, logging, math, os, queue, threading, time
import multiprocessing as mp
from fractions import Fraction
import numpy as np
import av
import pyarrow.parquet as pq

NAMES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
CLASSES = list(NAMES)


def rot_ccw(stream, frame):
    r = (stream.metadata or {}).get("rotate")
    if r is not None:
        try:
            return (-int(float(r))) % 360
        except ValueError:
            pass
    try:
        return int(getattr(frame, "rotation", 0) or 0) % 360
    except Exception:
        return 0


def infer(model, imgs, dev):
    res = model.predict(imgs, imgsz=640, conf=0.4, iou=0.7, classes=CLASSES,
                        half=True, device=dev, verbose=False)
    return [r.boxes.cls.cpu().numpy().astype(int) for r in res]


def summarize(vid, per_sec, last_sec):
    for s in range(last_sec + 1):
        fr = per_sec.get(s, [])
        n = len(fr)
        if n == 0:
            objs, people = [], 0
        else:
            objs = [nm for cid, nm in NAMES.items()
                    if 2 * sum(1 for c in fr if cid in c) >= n]
            counts = sorted(int((c == 0).sum()) for c in fr)
            people = counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) // 2
        yield {"video": vid, "second": s, "objects": objs, "people": people}


def _producer(fp, name, a, q):
    pf = pq.ParquetFile(fp)
    row = 0
    for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
        if a.max_videos and row >= a.max_videos:
            break
        vid = f"{name}:{row}"
        row += 1
        imgs, secs, seen = [], [], set()
        last_sec, t0, idx, err = -1, None, 0, None
        try:
            mp4 = batch.column(0)[0].as_py()
            cont = av.open(io.BytesIO(mp4))
            st = cont.streams.video[0]
            st.thread_type = "AUTO"
            fps = float(st.average_rate or 30)
            stride = max(1, round(fps / a.sps))
            for f in cont.decode(st):
                if f.pts is not None:
                    t = Fraction(f.pts) * st.time_base
                else:
                    t = Fraction(idx) / Fraction(fps).limit_denominator(1000)
                if t0 is None:
                    t0 = t
                sec = max(math.floor(t - t0), 0)
                last_sec = max(last_sec, sec)
                if idx % stride == 0 or sec not in seen:
                    seen.add(sec)
                    img = f.to_ndarray(format="bgr24")
                    k = (rot_ccw(st, f) // 90) % 4
                    if k:
                        img = np.ascontiguousarray(np.rot90(img, k))
                    imgs.append(img); secs.append(sec)
                    if len(imgs) >= a.batch:
                        q.put(("f", vid, secs, imgs)); imgs, secs = [], []
                idx += 1
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:100]}"
        if imgs:
            q.put(("f", vid, secs, imgs))
        q.put(("end", vid, last_sec, err))


def producer(fp, name, a, q):
    try:
        _producer(fp, name, a, q)
    except Exception as e:
        print("producer failed:", e, flush=True)
    finally:
        q.put(("done", None, None, None))


def run_file(model, fp, a, dev, part):
    name = os.path.basename(fp)
    q = queue.Queue(maxsize=6)
    threading.Thread(target=producer, args=(fp, name, a, q), daemon=True).start()
    cur, ierr, nvid, ninf = {}, None, 0, 0
    with open(part + ".tmp", "w") as out:
        while True:
            kind, x, y, z = q.get()
            if kind == "done":
                break
            if kind == "f":
                if ierr:
                    continue
                try:
                    outs = infer(model, z, dev)
                    for s, c in zip(y, outs):
                        cur.setdefault(s, []).append(c)
                    ninf += len(z)
                except Exception as e:
                    ierr = f"{type(e).__name__}: {str(e)[:100]}"
            else:
                lines = list(summarize(x, cur, y)) if y >= 0 else []
                if z or ierr or y < 0:
                    lines.append({"video": x, "error": z or ierr or "could not decode video"})
                for l in lines:
                    out.write(json.dumps(l) + "\n")
                cur, ierr = {}, None
                nvid += 1
    os.replace(part + ".tmp", part)
    return nvid, ninf


def worker(files, dev, a):
    from ultralytics import YOLO
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11l.pt"))
    parts = a.output + ".parts"
    for fp in files:
        name = os.path.basename(fp)
        part = os.path.join(parts, name + ".jsonl")
        if os.path.exists(part):
            print(f"[gpu{dev}] skip {name}", flush=True)
            continue
        t = time.time()
        nvid, ninf = run_file(model, fp, a, dev, part)
        print(f"[gpu{dev}] {name}: {nvid} videos, {ninf} frames, {time.time()-t:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sps", type=float, default=6, help="sampled frames per second of video")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=0, help="per file, for testing")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    import torch
    ng = a.gpus or max(1, torch.cuda.device_count())
    os.makedirs(a.output + ".parts", exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input, "*.parquet")))
    files = files[a.start:(a.end or None)]
    T0 = time.time()
    if ng == 1:
        worker(files, 0, a)
    else:
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=worker, args=(files[i::ng], i, a)) for i in range(ng)]
        for p in ps: p.start()
        for p in ps: p.join()
    with open(a.output, "w") as out:
        for fp in files:
            p = os.path.join(a.output + ".parts", os.path.basename(fp) + ".jsonl")
            if os.path.exists(p):
                out.write(open(p).read())
    print(f"TOTAL {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
