"""Keep long training jobs alive without letting them take the machine down.

    python scripts/watchdog.py JOBS.json

JOBS.json:
    {"max_jobs": 3,                       # jobs running at once
     "jobs": [{"name": "lr0.001 plain",
               "cwd": "/path/to/repo",
               "cmd": ["/opt/anaconda3/bin/python", "-u", "-m", "src.train.packets", ..., "--resume"],
               "done": "runs/x/time_mmcfc/results.json",   # job is finished when this exists
               "watch": "runs/x/time_mmcfc",               # newest file here is the heartbeat
               "log": "runs/x/resume_plain.log"}]}

What it does, every POLL seconds:
- Starts queued jobs while fewer than max_jobs run, unless memory is short or the battery is low.
- A job that exits without its done file is relaunched (its command should use --resume),
  up to MAX_RESTARTS times. If its log shows a non-finite loss it is marked diverged and
  not retried, because resuming from the same checkpoint repeats the crash.
- A job whose watch folder hasn't changed for STALL_MIN minutes is killed and relaunched.
- If free memory falls below MEM_CRITICAL (or swap is above SWAP_CRITICAL while free memory is
  below MEM_LOW), the most recently started job is stopped and queued again, so the machine
  never thrashes into a hang. macOS keeps swap allocated after pressure passes, so swap alone
  is not treated as a problem.
- On battery at or below BATTERY_STOP percent every job is stopped (they resume from their
  last epoch checkpoint); jobs start again once the Mac is on AC power.
Status goes to stdout and to watchdog_status.json beside JOBS.json. Stop it with Ctrl-C;
running jobs are stopped too and can resume later.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL = 30
STALL_MIN = 20
MAX_RESTARTS = 5
MEM_LOW, MEM_CRITICAL = 20, 10          # percent of memory free
SWAP_CRITICAL = 0.85                    # share of swap in use
BATTERY_STOP = 15
DIVERGED = re.compile(r"non-finite|NaN loss", re.I)


def memory_free_percent():
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)", out)
    return int(m.group(1)) if m else 100


def swap_share():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    total, used = (float(re.search(rf"{k} = ([\d.]+)M", out).group(1)) for k in ("total", "used"))
    return used / total if total else 0.0


def battery():
    """(percent, on_ac)."""
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%", out)
    return (int(m.group(1)) if m else 100), "AC Power" in out


def newest_change(folder):
    files = [p for p in Path(folder).rglob("*") if p.is_file()]
    return max((p.stat().st_mtime for p in files), default=0.0)


class Job:
    def __init__(self, spec):
        self.spec, self.name = spec, spec["name"]
        self.cwd = Path(spec["cwd"])
        self.proc, self.started, self.restarts, self.state = None, 0.0, 0, "queued"

    def path(self, key):
        return self.cwd / self.spec[key]

    def finished(self):
        return self.path("done").exists()

    def diverged(self):
        log = self.path("log")
        return log.exists() and bool(DIVERGED.search(log.read_text(errors="ignore")[-20000:]))

    def start(self):
        self.path("log").parent.mkdir(parents=True, exist_ok=True)
        out = open(self.path("log"), "a")
        out.write(f"\n=== watchdog start {time.ctime()} (restart {self.restarts}) ===\n")
        out.flush()
        self.proc = subprocess.Popen(["caffeinate", "-i", *self.spec["cmd"]], cwd=self.cwd,
                                     stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.started, self.state = time.time(), "running"

    def stop(self, state="queued"):
        if self.proc and self.proc.poll() is None:
            # The whole process group: the job may have spawned seed workers.
            try:

                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.proc, self.state = None, state


def main():
    jobs_file = Path(sys.argv[1]).resolve()
    config = json.loads(jobs_file.read_text())
    jobs = [Job(s) for s in config["jobs"]]
    max_jobs = config.get("max_jobs", 2)
    status_file = jobs_file.with_name("watchdog_status.json")
    stopping = False

    def on_signal(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    while not stopping:
        free, swap = memory_free_percent(), swap_share()
        percent, on_ac = battery()
        notes = []
        for job in jobs:
            if job.state == "running" and job.proc.poll() is not None:
                code = job.proc.returncode
                if job.finished():
                    job.state = "done"
                elif job.diverged():
                    job.state = "diverged"
                elif job.restarts >= MAX_RESTARTS:
                    job.state = "failed"
                else:
                    job.restarts += 1
                    job.state = "queued"
                notes.append(f"{job.name} exited ({code}) -> {job.state}")
            elif job.state == "running" and time.time() - max(job.started, newest_change(job.path("watch"))) > STALL_MIN * 60:
                job.restarts += 1
                job.stop()
                notes.append(f"{job.name} stalled for {STALL_MIN} min, restarting")
        running = [j for j in jobs if j.state == "running"]
        if not on_ac and percent <= BATTERY_STOP:
            for job in running:
                job.stop()
            if running:
                notes.append(f"battery {percent}%: stopped {len(running)} jobs until AC power")
        elif free < MEM_CRITICAL or (swap > SWAP_CRITICAL and free < MEM_LOW):
            if running:
                newest = max(running, key=lambda j: j.started)
                newest.stop()
                notes.append(f"memory free {free}%, swap {swap:.0%}: stopped {newest.name}")
        elif free >= MEM_LOW and (on_ac or percent > BATTERY_STOP):
            for job in jobs:
                if job.state == "queued" and len([j for j in jobs if j.state == "running"]) < max_jobs:
                    if job.finished():
                        job.state = "done"
                        continue
                    job.start()
                    notes.append(f"started {job.name}")
        summary = {j.name: {"state": j.state, "restarts": j.restarts} for j in jobs}
        status_file.write_text(json.dumps({"time": time.ctime(), "memory_free": free, "swap": round(swap, 2),
                                           "battery": percent, "on_ac": on_ac, "jobs": summary}, indent=1) + "\n")
        for note in notes:
            print(f"{time.strftime('%H:%M:%S')} {note}", flush=True)
        if all(j.state in ("done", "diverged", "failed") for j in jobs):
            break
        time.sleep(POLL)
    for job in jobs:
        job.stop(job.state if job.state != "running" else "queued")
    print("final:", json.dumps({j.name: j.state for j in jobs}), flush=True)


if __name__ == "__main__":
    main()
