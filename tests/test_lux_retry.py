"""The Lux pair training job (scripts/lux/train_pair.sbatch) and its rerun loop (retry.sh).

Runs the bash function with fake commands: a run that fails twice then succeeds (as a transient
DataLoader bus error would), one that always fails, one that fails only after running a while, one
stopped by a signal, and a TERM sent to the loop itself while its command runs. Then the whole job
script, with fake `uv` and `nvidia-smi`, on every bash here (macOS /bin/bash is 3.2, older than
Lux's 4.2): reruns, the 2-GPU check, an arm out of retries, and scancel.
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RETRY = ROOT / "scripts" / "lux" / "retry.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def run_loop(tmp_path, command, retries=5, pause=0, env=None):
    """Run `run_with_retries` in a subshell under the job's shell options; (status, log, stdout)."""
    log = tmp_path / "arm.log"
    script = (
        f"set -euo pipefail\nsource '{RETRY}'\n"
        f"( run_with_retries '{log}' {retries} {pause} bash -c '{command}' ) &\n"
        'status=0\nwait "$!" || status=$?\nexit "$status"\n'
    )
    proc = subprocess.run(
        [BASH, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **(env or {})},
    )
    return proc.returncode, log.read_text(), proc.stdout


def runs(log):
    """The fake command's own "run N" lines in the log."""
    return [line for line in log.splitlines() if line.startswith("run ")]


def gone(pid, wait=5.0):
    """True once no process `pid` exists (killed and reaped)."""
    for _ in range(int(wait / 0.05)):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def counter(tmp_path, fail_times, code=1, seconds=0):
    """A command that runs `seconds`, then fails `fail_times` times with `code`, then succeeds."""
    count = tmp_path / "count"
    return (
        f'n=$(( $(cat "{count}" 2>/dev/null || echo 0) + 1 )); echo $n > "{count}"; echo "run $n"; '
        f"sleep {seconds}; (( n > {fail_times} )) || exit {code}"
    )


def test_a_command_that_fails_twice_is_rerun_until_it_succeeds(tmp_path):
    status, log, stdout = run_loop(tmp_path, counter(tmp_path, 2))
    assert status == 0
    assert runs(log) == ["run 1", "run 2", "run 3"]
    assert "attempt 1: bash -c" in log and "attempt 2 failed (exit 1) after 0 s; rerun 2/5" in log
    assert "attempt 3 succeeded" in log
    assert "attempt 3 succeeded" in stdout  # also in the job's own output


def test_reruns_stop_after_retries_and_return_the_last_status(tmp_path):
    status, log, _ = run_loop(tmp_path, counter(tmp_path, 99, code=7), retries=2)
    assert status == 7
    assert len(runs(log)) == 3  # the first attempt and 2 reruns
    assert "attempt 3 failed (exit 7) after 0 s; no retries left (2 in a row)" in log


def test_an_attempt_that_ran_long_before_failing_starts_the_count_again(tmp_path):
    """Crashes days apart in a two-week run are each a fresh transient: with RETRY_RESET = 2 s,
    attempts that ran 2.5 s are rerun past RETRIES, while quick failures still stop."""
    slow, quick = tmp_path / "slow", tmp_path / "quick"
    slow.mkdir()
    quick.mkdir()
    reset = {"RETRY_RESET": "2"}
    status, log, _ = run_loop(slow, counter(slow, 3, seconds=2.5), retries=1, env=reset)
    assert status == 0 and len(runs(log)) == 4  # 3 failures with RETRIES = 1
    assert log.count("count reset") == 3
    status, log, _ = run_loop(quick, counter(quick, 3), retries=1, env=reset)
    assert status == 1 and len(runs(log)) == 2 and "count reset" not in log


@pytest.mark.parametrize("code", [130, 143])
def test_a_run_stopped_by_a_signal_is_not_rerun(tmp_path, code):
    status, log, _ = run_loop(tmp_path, counter(tmp_path, 99, code=code))
    assert status == code
    assert runs(log) == ["run 1"] and "attempt 1 was stopped" in log


def test_term_stops_the_running_command_and_its_children_without_a_rerun(tmp_path):
    log = tmp_path / "arm.log"
    pid_file = tmp_path / "child.pid"
    # like `uv run python ...`: the command's own child is the long-running process
    command = f'sleep 300 & echo $! > "{pid_file}"; wait'
    script = (
        f"set -euo pipefail\nsource '{RETRY}'\nrun_with_retries '{log}' 5 0 bash -c '{command}'"
    )
    loop = subprocess.Popen([BASH, "-c", script], stdout=subprocess.DEVNULL)
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.05)
    child = int(pid_file.read_text())
    loop.send_signal(signal.SIGTERM)
    assert loop.wait(timeout=20) == 143
    assert gone(child)  # the command's child (python under uv run) is stopped too
    text = log.read_text()
    assert "stopped by TERM/INT; not retrying" in text and "attempt 2" not in text


def test_pair_job_script_parses_and_defaults_to_the_two_arms():
    subprocess.run([BASH, "-n", str(ROOT / "scripts/lux/train_pair.sbatch")], check=True)
    subprocess.run([BASH, "-n", str(RETRY)], check=True)
    text = (ROOT / "scripts/lux/train_pair.sbatch").read_text()
    assert 'CONFIG_A="${CONFIG_A:-neo/configs/lux_neo_relu.ini}"' in text
    assert 'CONFIG_B="${CONFIG_B:-neo/configs/lux_neo_lrtrunk.ini}"' in text
    assert "#SBATCH --time=21-00:00:00" in text
    for config in ("lux_neo_relu.ini", "lux_neo_lrtrunk.ini"):
        assert (ROOT / "neo" / "configs" / config).is_file()


# The whole pair job, in a copy of scripts/lux with fake `uv` and `nvidia-smi` on PATH
LUX = ROOT / "scripts" / "lux"
BASHES = sorted({b for b in ("/bin/bash", BASH) if b and Path(b).is_file()})
FAKE_UV = r"""#!/bin/bash
# fake `uv run python ...`: the manifest guard passes; `train.py CONFIG --resume` fails
# FAKE_FAILS_<config name> times with a bus error and then succeeds, or with FAKE_HANG runs until
# stopped, like python under uv run (a child process)
shift 2
if [[ "$1" == "-m" ]]; then echo "fake guard $4"; exit 0; fi
cfg=$(basename "$2" .ini)
count="$FAKE_STATE/$cfg.count"
n=$(( $(cat "$count" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$count"
echo "fake train $cfg attempt $n on GPU $CUDA_VISIBLE_DEVICES"
fails="FAKE_FAILS_$cfg"
if (( n <= ${!fails:-0} )); then echo "Bus error (fake)"; exit 1; fi
if [[ -n "${FAKE_HANG:-}" ]]; then sleep 300 & echo $! > "$FAKE_STATE/$cfg.python"; wait; fi
exit 0
"""
FAKE_NVIDIA_SMI = r"""#!/bin/bash
if [[ "$1" == "-L" ]]; then
  for i in $(seq 1 "${FAKE_GPUS:-2}"); do echo "GPU $((i - 1)): Tesla V100-SXM2-32GB (fake)"; done
else
  echo "index, name"
fi
"""


class PairJob:
    """train_pair.sbatch run as Slurm would (from the submit dir), in its own process group so
    that a TERM to the group reaches every process of the job at once, as scancel does. Its output
    goes to a file: job_tmp.sh's cleaner leaves its `sleep` behind (Slurm's cgroup ends it on
    Lux), which would hold a pipe open; the fixture kills the group afterwards."""

    def __init__(self, tmp_path, bash, **env):
        self.repo, self.state, self.data = tmp_path / "repo", tmp_path / "state", tmp_path / "data"
        (self.repo / "scripts" / "lux").mkdir(parents=True)
        (self.repo / "logs").mkdir()
        self.state.mkdir()
        for name in ("job_tmp.sh", "comet_check.sh", "retry.sh", "train_pair.sbatch"):
            shutil.copy(LUX / name, self.repo / "scripts" / "lux" / name)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, text in (("uv", FAKE_UV), ("nvidia-smi", FAKE_NVIDIA_SMI)):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        job_env = {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "SLURM_SUBMIT_DIR": str(self.repo),
            "SLURM_JOB_ID": "4242",
            "SLURM_JOB_NAME": "neo-train-pair",
            "NEO_DATA": str(self.data),
            "COMET_OFFLINE": "1",
            "FAKE_STATE": str(self.state),
            "CONFIG_A": "neo/configs/a.ini",
            "CONFIG_B": "neo/configs/b.ini",
            "RETRY_PAUSE": "0",
            "JOB_TMP_SWEEP_SECONDS": "5",
            **env,
        }
        self.out = tmp_path / "job.out"
        with open(self.out, "w") as out:
            self.proc = subprocess.Popen(
                [bash, "scripts/lux/train_pair.sbatch"],
                cwd=self.repo,
                env=job_env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def wait(self):
        """(exit status, the job's own output)."""
        status = self.proc.wait(timeout=60)
        return status, self.out.read_text()

    def log(self, arm):
        path = self.repo / "logs" / f"train-{arm}-4242.out"
        return path.read_text() if path.exists() else ""

    def tmpdirs(self):
        return list((self.data / "tmp").iterdir())

    def python(self, arm, wait=10.0):
        """pid of an arm's fake python, once it runs."""
        path = self.state / f"{arm}.python"
        for _ in range(int(wait / 0.05)):
            if path.exists() and path.read_text().strip():
                return int(path.read_text())
            time.sleep(0.05)
        raise AssertionError(f"arm {arm} never started")

    def kill(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.fixture(params=BASHES)
def pair_job(request, tmp_path):
    jobs = []

    def start(**env):
        jobs.append(PairJob(tmp_path, request.param, **env))
        return jobs[-1]

    yield start
    for job in jobs:  # nothing of a failed test keeps running
        job.kill()


def test_pair_job_reruns_a_crashed_arm_on_its_own_gpu_and_waits_for_both(pair_job):
    job = pair_job(FAKE_FAILS_a="2")
    status, out = job.wait()
    assert status == 0, out
    a, b = job.log("a"), job.log("b")
    assert [line for line in a.splitlines() if line.startswith("fake train")] == [
        f"fake train a attempt {n} on GPU 0" for n in (1, 2, 3)
    ]
    assert "attempt 3 succeeded" in a and "Bus error" in a
    assert "fake train b attempt 1 on GPU 1" in b and "attempt 2" not in b
    assert "exit status: neo/configs/a.ini 0, neo/configs/b.ini 0" in out
    assert job.tmpdirs() == []  # TMPDIR removed at exit


def test_pair_job_refuses_a_node_with_one_gpu(pair_job):
    job = pair_job(FAKE_GPUS="1")
    status, out = job.wait()
    assert status == 1 and "need 2 GPUs" in out
    assert job.log("a") == job.log("b") == ""  # neither arm started


def test_pair_job_refuses_the_same_config_twice(pair_job):
    status, out = pair_job(CONFIG_B="neo/configs/a.ini").wait()
    assert status == 1 and "same file" in out


def test_an_arm_out_of_retries_fails_the_job_but_the_other_arm_finishes(pair_job):
    job = pair_job(FAKE_FAILS_b="99", RETRIES="1")
    status, out = job.wait()
    assert status == 1
    assert "attempt 1 succeeded" in job.log("a")
    assert job.log("b").count("fake train b") == 2 and "no retries left" in job.log("b")
    assert "exit status: neo/configs/a.ini 0, neo/configs/b.ini 1" in out


@pytest.mark.parametrize("scancel", [True, False], ids=["term-to-every-process", "term-to-job"])
def test_term_stops_both_arms_without_reruns_and_removes_tmpdir(pair_job, scancel):
    """scancel signals every process of the job at once: the job_tmp.sh cleaner is then gone
    before the EXIT trap kills it, which (without `|| true`) used to end the trap before rm and
    turn exit 143 into 1. A TERM to the job script alone must reach both pythons via the loops."""
    job = pair_job(FAKE_HANG="1")
    pythons = [job.python("a"), job.python("b")]
    assert len(job.tmpdirs()) == 1
    if scancel:
        os.killpg(job.proc.pid, signal.SIGTERM)
    else:
        job.proc.send_signal(signal.SIGTERM)
    status, out = job.wait()
    assert status == 143, out
    assert all(gone(pid) for pid in pythons)
    assert job.tmpdirs() == []
    for arm in ("a", "b"):
        assert "attempt 2" not in job.log(arm)
