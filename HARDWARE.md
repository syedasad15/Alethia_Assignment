# Hardware

All 50 files were processed on one machine, a Kaggle notebook, in two sequential runs
(files 0-32, then files 33-49), both with `python run.py --sps 6`.

- GPU: 2x NVIDIA Tesla T4 (15 GB each), driver 580.178.04
- CPU: Intel Xeon @ 2.00 GHz, 4 vCPUs (2 cores x 2 threads)
- RAM: 31 GB
- Disk: Google "PersistentDisk" network storage (about 8 TB, shared with other containers). Linux reports it as non-rotational (SSD-class). The videos were on `/kaggle/temp` (container filesystem) and the results on `/kaggle/working` (20 GB ext4)
- OS: Ubuntu 24.04, Python 3.13 (the grading machine uses Python 3.11)
- Run time: files 0-32 took 10,051 s (2.8 h, started 15:10 PKT); files 33-49 took 5,081 s (1.4 h, started 18:12 PKT). About 4.2 h in total, with the downloads in addition.

I first started files 33-49 on Google Colab (1x T4, 2 vCPUs), but that session was cut off and none of its output is used.
The T4s are much slower than the 4090 on the grading machine.
