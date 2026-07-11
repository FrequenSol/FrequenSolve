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
scheduler_status="$dir_out/scheduler_status.json"

ml intel/25.1 phdf5 petsc/3.23 fftw3
module list

export FS_SOLVER_PATH={{fs_dir}}
export KMP_STACKSIZE=24M
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

mpi_exec={{mpi}}
n_procs={{n_procs}}
n_threads={{n_threads}}
n_tasks={{n_tasks}}
n_job_tasks={{n_job_tasks}}
task_indices_json='{{task_indices_json}}'
executable={{executable}}
fresh_flag=""
{% if fresh %}
fresh_flag="--fresh"
{% endif %}
cat > "$scheduler_status" <<EOF
{"state":"pending","total":$n_tasks,"successful":0,"failed":0,"running":0,"pending":$n_tasks,"complete":0}
EOF

mark_scheduler_failed() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        python3 - "$scheduler_status" "$n_tasks" <<'PY' || true
import json, os, sys, time

status_file = sys.argv[1]
n_tasks = int(sys.argv[2])
try:
    with open(status_file, "r") as f:
        payload = json.load(f)
except Exception:
    payload = {"total": n_tasks, "successful": 0, "failed": 0, "running": 0, "pending": n_tasks, "complete": 0}
payload["state"] = "failed"
payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
tmp = f"{status_file}.tmp"
with open(tmp, "w") as f:
    json.dump(payload, f, separators=(",", ":"))
    f.write("\n")
os.replace(tmp, status_file)
PY
    fi
}
trap mark_scheduler_failed EXIT
TOTAL_RANKS={{n_procs}}
MEM_PER_RANK_GIB={{proc_memory}}
MIN_RANKS={{min_ranks}}
ROUND_TO={{round_to}}
CAP_FRACTION={{cap_fraction}}
MEM_CUSHION={{mem_cushion}}
BOOST_MAX_FACTOR={{boost_max_factor}}
FAILURE_TOLERANCE={{tolerate_failures}}
FS_SKIP_SIZING={{skip_sizing}}
{% if sizing_json %}
FS_SIZING_JSON="{{sizing_json}}"
{% else %}
FS_SIZING_JSON="FS_sizing.json"
{% endif %}
export FS_SIZING_JSON="$FS_SIZING_JSON"
export FS_SKIP_SIZING="$FS_SKIP_SIZING"

validate_sizing_checkpoint() {
    python3 - "$FS_SIZING_JSON" "$n_job_tasks" <<'PY'
import json, sys

path = sys.argv[1]
n_tasks = int(sys.argv[2])

with open(path, "r") as f:
    payload = json.load(f)

if payload.get("schema") != "fs-sizing-2":
    raise SystemExit(f"invalid sizing schema in {path}")

status = payload.get("sweep_status", "complete")
if status not in ("forward_sweep_checkpoint", "complete"):
    raise SystemExit(f"invalid sizing sweep_status={status!r}")

tasks = payload.get("task")
if not isinstance(tasks, list) or len(tasks) < n_tasks:
    raise SystemExit(f"missing task estimates in {path}")

for i, task in enumerate(tasks[:n_tasks]):
    if int(task.get("memory_bytes", 0)) <= 0:
        raise SystemExit(f"task {i + 1} missing memory estimate")
PY
}

start_time=$(date +%s)
{% if not smooth_only %}
rm -f "$FS_SIZING_JSON"
set +e
if [ "$FS_SKIP_SIZING" = "1" ]; then
    echo "$mpi_exec -n $n_procs $executable -nthreads $n_threads --job $job_file $fresh_flag --init-no-size --map"
    $mpi_exec -n $n_procs $executable -nthreads $n_threads --job $job_file $fresh_flag --init-no-size --map
else
    echo "$mpi_exec -n $n_procs $executable -nthreads $n_threads --job $job_file $fresh_flag --init --map"
    $mpi_exec -n $n_procs $executable -nthreads $n_threads --job $job_file $fresh_flag --init --map
fi
sizing_rc=$?
set -e
if [ "$sizing_rc" -ne 0 ]; then
    if [ "$FS_SKIP_SIZING" != "1" ] && validate_sizing_checkpoint; then
        echo "[scheduler] sizing exited with $sizing_rc after writing usable estimates; continuing with $FS_SIZING_JSON"
    else
        exit "$sizing_rc"
    fi
fi

export FS_JOB_FILE="$job_file"
export DIR_OUT="$dir_out"
export EXE="$executable"
export MPI_EXEC="$mpi_exec"
export FS_FRESH_FLAG="$fresh_flag"
export FS_SCHEDULER_STATUS="$scheduler_status"
export FS_TASK_INDICES="$task_indices_json"
export FS_JOB_TASK_COUNT="$n_job_tasks"
export TOTAL_RANKS="$TOTAL_RANKS"
export OMP_THREADS="$n_threads"
export MEM_PER_RANK_GIB="$MEM_PER_RANK_GIB"
export MIN_RANKS="$MIN_RANKS"
export ROUND_TO="$ROUND_TO"
export CAP_FRACTION="$CAP_FRACTION"
export MEM_CUSHION="$MEM_CUSHION"
export BOOST_MAX_FACTOR="$BOOST_MAX_FACTOR"
export FAILURE_TOLERANCE="$FAILURE_TOLERANCE"

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
fresh_flag    = os.environ.get("FS_FRESH_FLAG", "").split()
status_file   = os.environ.get("FS_SCHEDULER_STATUS")
total_ranks   = int(os.environ.get("TOTAL_RANKS", "1"))
omp_threads   = int(os.environ.get("OMP_THREADS", "1"))
mem_per_rank  = float(os.environ["MEM_PER_RANK_GIB"])  # GiB per MPI rank from driver
job_task_count   = int(os.environ.get("FS_JOB_TASK_COUNT", "0") or "0")
task_indices_raw = os.environ.get("FS_TASK_INDICES", "").strip()
skip_sizing      = os.environ.get("FS_SKIP_SIZING", "0").strip().lower() in {"1", "true", "yes", "on"}

MIN_RANKS        = int(os.environ.get("MIN_RANKS", "1"))
ROUND_TO         = int(os.environ.get("ROUND_TO", "2"))
CAP_FRACTION     = float(os.environ.get("CAP_FRACTION", "1.0"))
MEM_CUSHION      = float(os.environ.get("MEM_CUSHION", "1.5"))
BOOST_MAX_FACTOR = float(os.environ.get("BOOST_MAX_FACTOR", "4.0"))
failure_tolerance_raw = os.environ.get("FAILURE_TOLERANCE", "4").strip().lower()
if failure_tolerance_raw in {"none", "unlimited", "inf", "infinite"}:
    FAILURE_TOLERANCE = None
else:
    FAILURE_TOLERANCE = int(failure_tolerance_raw)

sizing_json = os.environ.get("FS_SIZING_JSON")
print(f"[scheduler] sizing_json={sizing_json}", flush=True)
if sizing_json is None:
    raise ValueError("FS_SIZING_JSON environment variable is not set")
sizing_json = sizing_json.strip()
print(f"[scheduler] sizing_json={sizing_json}", flush=True)

max_ranks_per_task = total_ranks if skip_sizing else max(1, int(total_ranks * CAP_FRACTION))

print(f"[scheduler] total_ranks={total_ranks} omp_threads={omp_threads}", flush=True)
print(f"[scheduler] mem_per_rank={mem_per_rank:.3f} GiB cushion={MEM_CUSHION}", flush=True)
print(f"[scheduler] min_ranks={MIN_RANKS} round_to={ROUND_TO} cap_fraction={CAP_FRACTION}", flush=True)
print(f"[scheduler] boost_max_factor={BOOST_MAX_FACTOR}", flush=True)
print(f"[scheduler] tolerate_failures={FAILURE_TOLERANCE}", flush=True)
print(f"[scheduler] skip_sizing={skip_sizing}", flush=True)

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
    if skip_sizing:
        count = job_task_count
        if count <= 0 and task_indices_raw:
            indices = [int(value) for value in json.loads(task_indices_raw)]
            count = max(indices, default=0)
        count = max(count, 1)
        return [0.0] * count

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

def write_status(
    state: str,
    *,
    running_tasks=None,
    successful_tasks=None,
    failed_tasks=None,
    aborted_tasks=None,
    abort_reason=None,
):
    if not status_file:
        return
    running_tasks = list(running_tasks or [])
    successful_tasks = list(successful_tasks or [])
    failed_tasks = list(failed_tasks or [])
    aborted_tasks = list(aborted_tasks or [])
    complete = len(successful_tasks) + len(failed_tasks)
    pending = max(0, n_tasks - complete - len(running_tasks))
    payload = {
        "state": state,
        "total": n_tasks,
        "successful": len(successful_tasks),
        "failed": len(failed_tasks),
        "running": len(running_tasks),
        "pending": pending,
        "complete": complete,
        "running_tasks": running_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "aborted_tasks": aborted_tasks,
        "tolerate_failures": FAILURE_TOLERANCE,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if abort_reason:
        payload["abort_reason"] = abort_reason
    tmp = f"{status_file}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, status_file)

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

def failure_tolerance_exceeded() -> bool:
    return FAILURE_TOLERANCE is not None and len(failed_tasks) > FAILURE_TOLERANCE

def abort_running_tasks(reason: str):
    aborted = []
    for p, offset, ranks, tid, m, out in running:
        aborted.append(tid)
        try:
            p.terminate()
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        finally:
            try:
                out.close()
            except Exception:
                pass
    write_status(
        "failed",
        running_tasks=[],
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        aborted_tasks=aborted,
        abort_reason=reason,
    )
    raise SystemExit(reason)

# --------------------------
# Load tasks
# --------------------------
mem_gib = load_task_mems()
if job_task_count <= 0:
    job_task_count = len(mem_gib)
if len(mem_gib) < job_task_count:
    raise SystemExit(
        f"missing task estimates in {sizing_json}: "
        f"found {len(mem_gib)}, expected {job_task_count}"
    )
if task_indices_raw:
    task_indices = [int(value) for value in json.loads(task_indices_raw)]
else:
    task_indices = list(range(1, job_task_count + 1))
bad_indices = [task for task in task_indices if task < 1 or task > job_task_count]
if bad_indices:
    raise SystemExit(f"invalid task indices: {bad_indices}")
n_tasks = len(task_indices)
if skip_sizing and n_tasks != 1:
    raise SystemExit("--init-no-size scheduling requires exactly one submitted task")
print(f"[scheduler] tasks={n_tasks} task_ids={task_indices}", flush=True)

# Largest-first reduces fragmentation
order = sorted(task_indices, key=lambda i: mem_gib[i - 1], reverse=True)
queue = deque(order)

# running entries: (proc, offset, ranks, task_id, mem_gib)
running = []
successful_tasks = []
failed_tasks = []

# Track actual free slot intervals in hostfile slot space
free_intervals = [(0, total_ranks)]
write_status("running", running_tasks=[], successful_tasks=[], failed_tasks=[])

def launch(task_id: int, offset: int, ranks: int):
    out_path = os.path.join(dir_out, f"task_{task_id}.log")
    cmd = [
        mpi_exec,
        "-n", str(ranks),
        "-o", str(offset),
        "task_affinity",
        exe,
        "-nthreads", str(omp_threads),
        "--job", job_file,
        *fresh_flag,
        "--task", str(task_id),
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
    write_status(
        "running",
        running_tasks=[entry[3] for entry in running],
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
    )

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
                if rc == 0:
                    successful_tasks.append(tid)
                else:
                    failed_tasks.append(tid)
                free_intervals = free_interval(free_intervals, offset, ranks)
                print(
                    f"[scheduler] finish task={tid} rc={rc} "
                    f"freed_offset={offset} freed_ranks={ranks} "
                    f"free={total_free_ranks(free_intervals)} intervals={free_intervals}",
                    flush=True,
                )
        running = still
        if failure_tolerance_exceeded():
            queue.clear()
            reason = (
                f"failure tolerance exceeded: {len(failed_tasks)} failed tasks "
                f"(tolerate_failures={FAILURE_TOLERANCE})"
            )
            print(f"[scheduler] {reason}", flush=True)
            abort_running_tasks(reason)
        write_status(
            "running" if queue or running else ("failed" if failed_tasks else "complete"),
            running_tasks=[entry[3] for entry in running],
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
        )
    elif not launched_any:
        write_status(
            "running",
            running_tasks=[entry[3] for entry in running],
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
        )
        time.sleep(1)

write_status(
    "failed" if failed_tasks else "complete",
    running_tasks=[],
    successful_tasks=successful_tasks,
    failed_tasks=failed_tasks,
)
print("[scheduler] all tasks done", flush=True)
PY
{% else %}
echo "Skipping frequency sweep; running imaging postprocess only."
{% endif %}

{% if imaging_job %}
echo "Running imaging step..."
"$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --smooth >> "$dir_out/smooth.log" 2>&1
{% if smooth_only %}
cat > "$scheduler_status" <<EOF
{"state":"complete","total":0,"successful":0,"failed":0,"running":0,"pending":0,"complete":0}
EOF
{% endif %}
{% endif %}

{% if pack_job %}
echo "Running packing step..."
"$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --pack >> "$dir_out/pack.log" 2>&1
{% endif %}

end_time=$(date +%s)
total_seconds=$((end_time - start_time))
hours=$((total_seconds / 3600))
minutes=$(( (total_seconds % 3600) / 60 ))
seconds=$((total_seconds % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Sweep Complete"
