# Write-up

## 1. Your timings (100 words max)

V1 processed 10 videos in 877 seconds with estimated 33–34 hours for all 50 files V2 processed all 50 files in about 4.2 hours using two T4 GPUs The biggest improvement was reducing inference to 6 frames per second instead of processing every frame Background decoding and two-GPU processing also helped reduce the time.

## 2. Your pipeline (150 words max)

The files have been split into two gpus one thread per gpu,each thread reads the bytes with pyarrow and then pyav decodes that from memory .every stride-th frame is kept ensurin at least one frame at a second.the frames are converted intp bgr and main thread runs the yolo11.results are aggregated per second.

## 3. Where the time goes (150 words max)

the inference took the largest time. most of the time the usage of cpu was among 70-100%.

## 4. Checking correctness (100 words max)

i compared the faster pipeline against an every-frame reference on eight videos, using compare.py. At 6 FPS, it achieved 0.985 F1, 0.957 exact people-count accuracy, and 0.998 within-one accuracy.

## 5. Working with AI (100 words max)

One AI suggestion was to use av==19.0.2 in the requirements file.

## 6. Next steps (100 words max)

I would improve GPU workload distribution because the second GPU sat idle for around 25 minutes during the first run. I would also increase decoding workers to use the grader's 32 CPU threads or explore GPU decoding. I would test more than eight videos, including rotated videos, which were missing from my original sample.
