#!/bin/bash

{% if batch_job %}
#SBATCH -J {{ name }}
{% if run_path %}
#SBATCH -o {{run_path}}/jobs/batch/job_%j.o
#SBATCH -e {{run_path}}/jobs/batch/job_%j.e
{% else %}
#SBATCH -o ./job_%j.o
#SBATCH -e ./job_%j.e
{% endif %}
#SBATCH -N {{ n_nodes }}
#SBATCH -n {{ n_procs }}
#SBATCH -p {{ queue }}
{% if account %}
#SBATCH -A {{ account }}
{% endif %}
{% if duration %}
#SBATCH -t {{ duration }}
{% endif %}
{% if notify_on %}
#SBATCH --mail-type={{ notify_on }}
{% endif %}
{% if notify_email %}
#SBATCH --mail-user={{ notify_email }}
{% endif %}
{% endif %}

set -euo pipefail

{% if job_json %}
job_file={{job_json}}
{% else %}
job_file=$1
{% endif %}

{% if run_path %}
cd {{run_path}}
{% endif %}

dir_out={{dir_out}}
rm -rf "$dir_out"
mkdir -p "$dir_out"

ml intel/25.1 phdf5 petsc/3.23 fftw3
module list

export FS_SOLVER_PATH={{fs_dir}}
export KMP_STACKSIZE=30M

mpi_exec={{mpi}}
n_procs={{n_procs}}
n_threads={{n_threads}}
n_tasks={{n_tasks}}
executable={{executable}}
TOTAL_RANKS={{n_procs}}
MEM_PER_RANK_GIB={{proc_memory}}
MIN_RANKS={{min_ranks}}
ROUND_TO={{round_to}}
CAP_FRACTION={{cap_fraction}}
MEM_CUSHION={{mem_cushion}}
BOOST_MAX_FACTOR={{boost_max_factor}}
{% if sizing_json %}
FS_SIZING_JSON="{{sizing_json}}"
{% else %}
FS_SIZING_JSON="FS_sizing.json"
{% endif %}
export FS_SIZING_JSON="$FS_SIZING_JSON"

start_time=$(date +%s)
echo "$mpi_exec -n $n_procs $executable -nthreads $n_threads -j $job_file --init"
$mpi_exec -n $n_procs $executable -nthreads $n_threads -j $job_file --init
$mpi_exec -n $n_procs $executable -nthreads $n_threads -j $job_file --size

export FS_JOB_FILE="$job_file"
export DIR_OUT="$dir_out"
export EXE="$executable"
export MPI_EXEC="$mpi_exec"
export TOTAL_RANKS="$TOTAL_RANKS"
export OMP_THREADS="$n_threads"
export MEM_PER_RANK_GIB="$MEM_PER_RANK_GIB"
export MIN_RANKS="$MIN_RANKS"
export ROUND_TO="$ROUND_TO"
export CAP_FRACTION="$CAP_FRACTION"
export MEM_CUSHION="$MEM_CUSHION"
export BOOST_MAX_FACTOR="$BOOST_MAX_FACTOR"

python3 - <<'PY'
import json, math, os, re, subprocess, time
from collections import deque

# --------------------------
# Read policy knobs / inputs
# --------------------------
job_file      = os.environ["FS_JOB_FILE"]
dir_out       = os.environ["DIR_OUT"]
exe           = os.environ["EXE"]
mpi_exec      = os.environ.get("MPI_EXEC", "ibrun")
total_ranks   = int(os.environ.get("TOTAL_RANKS", "1"))
omp_threads   = int(os.environ.get("OMP_THREADS", "1"))
mem_per_rank  = float(os.environ["MEM_PER_RANK_GIB"])  # GiB per MPI rank from driver

MIN_RANKS        = int(os.environ.get("MIN_RANKS", "1"))
ROUND_TO         = int(os.environ.get("ROUND_TO", "2"))
CAP_FRACTION     = float(os.environ.get("CAP_FRACTION", "1.0"))
MEM_CUSHION      = float(os.environ.get("MEM_CUSHION", "1.5"))
BOOST_MAX_FACTOR = float(os.environ.get("BOOST_MAX_FACTOR", "4.0"))

sizing_json = os.environ.get("FS_SIZING_JSON")
print(f"[scheduler] sizing_json={sizing_json}", flush=True)
if sizing_json is None:
    raise ValueError("FS_SIZING_JSON environment variable is not set")
sizing_json = sizing_json.strip()
print(f"[scheduler] sizing_json={sizing_json}", flush=True)

max_ranks_per_task = max(1, int(total_ranks * CAP_FRACTION))

print(f"[scheduler] total_ranks={total_ranks} omp_threads={omp_threads}", flush=True)
print(f"[scheduler] mem_per_rank={mem_per_rank:.3f} GiB cushion={MEM_CUSHION}", flush=True)
print(f"[scheduler] min_ranks={MIN_RANKS} round_to={ROUND_TO} cap_fraction={CAP_FRACTION}", flush=True)
print(f"[scheduler] boost_max_factor={BOOST_MAX_FACTOR}", flush=True)

# --------------------------
# Helpers
# --------------------------
def parse_mem_to_gib(s: str) -> float:
    s = s.strip()
    m = re.match(r"([0-9]*\.?[0-9]+)\s*(KB|MB|GB|TB)", s, re.I)
    if not m:
        raise ValueError(f"Unrecognized memory string: {s!r}")
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "KB":
        return val / (1024 ** 2)
    if unit == "MB":
        return val / 1024
    if unit == "GB":
        return val
    if unit == "TB":
        return val * 1024
    raise ValueError(unit)

def round_up(x: int, base: int) -> int:
    if base <= 1:
        return x
    return int(math.ceil(x / base) * base)

def choose_base_ranks(task_mem_gib: float) -> int:
    r = int(math.ceil(task_mem_gib * MEM_CUSHION / mem_per_rank))
    r = max(r, MIN_RANKS)
    r = min(r, max_ranks_per_task, total_ranks)
    r = round_up(r, ROUND_TO)
    r = max(r, MIN_RANKS)
    r = min(r, total_ranks)
    return r

def load_task_mems():
    path = sizing_json if sizing_json else job_file
    with open(path, "r") as f:
        data = json.load(f)
    tasks = data.get("task")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit(
            f"Could not find non-empty JSON['task'] list in {path}. "
            "If sizing output is in a different file, set FS_SIZING_JSON."
        )
    mems = []
    for t in tasks:
        mems.append(parse_mem_to_gib(t["memory"]))
    return mems

def total_free_ranks(intervals):
    return sum(length for _, length in intervals)

def allocate_interval(intervals, ranks: int):
    """
    First-fit allocation from a list of (start, length) free intervals.
    Returns (offset, new_intervals) or (None, intervals) if no contiguous fit exists.
    """
    for i, (start, length) in enumerate(intervals):
        if length >= ranks:
            offset = start
            new_intervals = list(intervals)
            if length == ranks:
                del new_intervals[i]
            else:
                new_intervals[i] = (start + ranks, length - ranks)
            return offset, new_intervals
    return None, intervals

def free_interval(intervals, offset: int, ranks: int):
    """
    Insert (offset, ranks) and coalesce adjacent intervals.
    """
    merged = []
    inserted = False
    new_start = offset
    new_end = offset + ranks

    for start, length in sorted(intervals + [(offset, ranks)]):
        end = start + length
        if not merged:
            merged.append([start, end])
            continue

        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])

    return [(start, end - start) for start, end in merged]

# --------------------------
# Load tasks
# --------------------------
mem_gib = load_task_mems()
n_tasks = len(mem_gib)
print(f"[scheduler] tasks={n_tasks}", flush=True)

# Largest-first reduces fragmentation
order = sorted(range(1, n_tasks + 1), key=lambda i: mem_gib[i - 1], reverse=True)
queue = deque(order)

# running entries: (proc, offset, ranks, task_id, mem_gib)
running = []

# Track actual free slot intervals in hostfile slot space
free_intervals = [(0, total_ranks)]

def launch(task_id: int, offset: int, ranks: int):
    out_path = os.path.join(dir_out, f"task_{task_id}.log")
    cmd = [
        mpi_exec,
        "-n", str(ranks),
        "-o", str(offset),
        "task_affinity",
        exe,
        "-nthreads", str(omp_threads),
        "-j", job_file,
        "-i", str(task_id),
    ]
    out = open(out_path, "ab", buffering=0)
    try:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    except Exception:
        out.close()
        raise

    m = mem_gib[task_id - 1]
    running.append((p, offset, ranks, task_id, m, out))
    print(
        f"[scheduler] launch task={task_id} mem≈{m:.2f}GiB "
        f"offset={offset} ranks={ranks} free={total_free_ranks(free_intervals)} "
        f"intervals={free_intervals}",
        flush=True,
    )
    time.sleep(float(os.environ.get("LAUNCH_DELAY_SEC", "0.25")))

# --------------------------
# Scheduling loop
# --------------------------
while queue or running:
    launched_any = False

    running_mems = [m for _, _, _, _, m, _ in running]
    free_ranks = total_free_ranks(free_intervals)

    cands = []
    remaining = len(queue) + len(running)
    remaining_mem_gib = (
        sum(mem_gib[tid - 1] for tid in queue) +
        sum(running_mems)
    )

    for tid in list(queue):
        base = choose_base_ranks(mem_gib[tid - 1])
        need_min = base

        if len(queue) == 1:
            need_max = free_ranks
        elif remaining < total_ranks:
            task_mem = mem_gib[tid - 1]
            if remaining_mem_gib > 0.0:
                share = int(math.ceil(total_ranks * (task_mem / remaining_mem_gib)))
            else:
                share = base

            boosted = max(base, share)
            boosted = min(boosted, int(math.ceil(max(2.0, BOOST_MAX_FACTOR) * base)))
            need_max = boosted
        else:
            need_max = base

        need_min = min(need_min, max_ranks_per_task, total_ranks)
        need_min = max(need_min, MIN_RANKS)
        need_min = min(need_min, total_ranks)

        need_max = min(need_max, max_ranks_per_task, total_ranks)
        need_max = max(need_max, need_min)
        need_max = min(need_max, total_ranks)

        cands.append((tid, need_min, need_max, mem_gib[tid - 1]))

    # Plan against interval state, not just a scalar rank count
    sim_intervals = list(free_intervals)
    plan = []  # list of (task_id, offset, alloc_ranks)
    chosen = set()

    # Phase A: boosted launches, best-fit by smallest leftover in a fitting interval
    boost_pool = sorted(cands, key=lambda x: (x[2], x[3]), reverse=True)

    while True:
        best = None
        best_metric = None

        for tid, need_min, need_max, m in boost_pool:
            if tid in chosen:
                continue

            offset, trial_intervals = allocate_interval(sim_intervals, need_max)
            if offset is None:
                continue

            leftover = total_free_ranks(trial_intervals)
            largest_gap = max((length for _, length in trial_intervals), default=0)
            metric = (leftover, largest_gap)

            if best is None or metric < best_metric:
                best = (tid, offset, need_max, trial_intervals, m)
                best_metric = metric

        if best is None:
            break

        tid, offset, alloc, trial_intervals, m = best
        plan.append((tid, offset, alloc))
        sim_intervals = trial_intervals
        chosen.add(tid)
        launched_any = True

    # Phase B: fill with minimum-size launches, preferring larger minimums first
    min_pool = sorted(
        [(tid, need_min) for (tid, need_min, need_max, m) in cands if tid not in chosen],
        key=lambda x: x[1],
        reverse=True,
    )

    for tid, need_min in min_pool:
        offset, trial_intervals = allocate_interval(sim_intervals, need_min)
        if offset is None:
            continue
        plan.append((tid, offset, need_min))
        sim_intervals = trial_intervals
        chosen.add(tid)
        launched_any = True

    # Execute plan
    if plan:
        for tid, _, _ in plan:
            queue.remove(tid)

        for tid, offset, alloc in plan:
            actual_offset, new_intervals = allocate_interval(free_intervals, alloc)
            if actual_offset is None:
                raise RuntimeError(
                    f"Internal scheduler error: planned task={tid} alloc={alloc} "
                    f"but no fitting interval remained. free_intervals={free_intervals}"
                )
            if actual_offset != offset:
                print(
                    f"[scheduler] note: planned offset {offset} changed to {actual_offset} "
                    f"for task={tid}",
                    flush=True,
                )
            free_intervals = new_intervals
            launch(tid, actual_offset, alloc)

    # Reap finished steps
    if running:
        time.sleep(1)
        still = []
        for p, offset, ranks, tid, m, out in running:
            rc = p.poll()
            if rc is None:
                still.append((p, offset, ranks, tid, m, out))
            else:
                out.close()
                free_intervals = free_interval(free_intervals, offset, ranks)
                print(
                    f"[scheduler] finish task={tid} rc={rc} "
                    f"freed_offset={offset} freed_ranks={ranks} "
                    f"free={total_free_ranks(free_intervals)} intervals={free_intervals}",
                    flush=True,
                )
        running = still
    elif not launched_any:
        time.sleep(1)

print("[scheduler] all tasks done", flush=True)
PY

{% if imaging_job %}
echo "Running imaging step..."
"$executable" -j "$job_file" --smooth
{% endif %}

{% if pack_job %}
echo "Running packing step..."
"$executable" -nthreads "$n_threads" -j "$job_file" --pack >> "$dir_out/pack.log" 2>&1
{% endif %}

end_time=$(date +%s)
total_seconds=$((end_time - start_time))
hours=$((total_seconds / 3600))
minutes=$(( (total_seconds % 3600) / 60 ))
seconds=$((total_seconds % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Sweep Complete"
