#!/bin/bash

{% if batch_job %}
#SBATCH -J {{ name }}
{% if run_path %}
#SBATCH -o {{dir_out}}/batch/job_%j.o
#SBATCH -e {{dir_out}}/batch/job_%j.e
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
cd {{run_path_shell}}
{% endif %}

dir_out={{dir_out_shell}}
mkdir -p "$dir_out/batch"
find "$dir_out" -mindepth 1 -maxdepth 1 ! -name batch -exec rm -rf -- {} +
scheduler_status="$dir_out/scheduler_status.json"
scheduler_config="$dir_out/scheduler_config.json"
sizing_json={{sizing_json_shell}}
skip_sizing={{skip_sizing}}

{% for line in runtime_setup %}
{{ line }}
{% endfor %}

mpi_exec={{mpi_shell}}
mpi_args=( {{mpi_args_shell}} )
n_procs={{n_procs}}
init_ranks={{init_ranks}}
n_threads={{n_threads}}
export OMP_NUM_THREADS=$n_threads
{% for line in mpi_async_progress_setup %}
{{ line }}
{% endfor %}
n_tasks={{n_tasks}}
n_job_tasks={{n_job_tasks}}
executable={{executable_shell}}
fresh_flag=""
{% if fresh %}
fresh_flag="--fresh"
{% endif %}

printf '%s\n' {{ scheduler_config_shell }} > "$scheduler_config"

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

{% if mpi_health_check_timeout %}
mpi_health_timeout={{mpi_health_check_timeout_shell}}
mpi_health_log="$dir_out/mpi_health_check.log"
mpi_health_status="$dir_out/mpi_health_status.json"
allocation_nodelist="${SLURM_JOB_NODELIST:-unknown}"
if allocation_nodes=$(scontrol show hostnames "$allocation_nodelist" 2>/dev/null | paste -sd, -); then
    if [ -z "$allocation_nodes" ]; then allocation_nodes="$allocation_nodelist"; fi
else
    allocation_nodes="$allocation_nodelist"
fi

echo "[scheduler] checking MPI startup on ${allocation_nodes} (limit ${mpi_health_timeout})"
mpi_health_started=$(date +%s)
set +e
"$mpi_exec" "${mpi_args[@]}" --time="$mpi_health_timeout" -n "$n_procs" \
    "$executable" --job "$job_file" --mpi-health-check > "$mpi_health_log" 2>&1
mpi_health_rc=$?
if [ "$mpi_health_rc" -eq 0 ]; then
    python3 - "$mpi_health_log" "$n_procs" <<'PY'
import json, sys

log_file = sys.argv[1]
expected_ranks = int(sys.argv[2])
health = None
with open(log_file, "r") as f:
    for line in f:
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            continue
        if payload.get("schema") == "fs-mpi-health-1":
            health = payload
if health is None:
    raise SystemExit("MPI health check did not emit its success record")
if health.get("status") != "ok" or health.get("ranks") != expected_ranks:
    raise SystemExit(f"invalid MPI health check result: {health}")
PY
    mpi_health_rc=$?
fi
set -e
mpi_health_elapsed=$(($(date +%s) - mpi_health_started))

python3 - "$mpi_health_status" "$scheduler_status" "$mpi_health_rc" \
    "$mpi_health_timeout" "${SLURM_JOB_ID:-unknown}" "$allocation_nodelist" \
    "$allocation_nodes" "$mpi_health_elapsed" <<'PY' || true
import json, os, sys, time

health_file, scheduler_file = sys.argv[1:3]
return_code = int(sys.argv[3])
record = {
    "schema": "fs-mpi-health-status-1",
    "state": "complete" if return_code == 0 else "failed",
    "return_code": return_code,
    "timeout": sys.argv[4],
    "job_id": sys.argv[5],
    "nodelist": sys.argv[6],
    "nodes": [node for node in sys.argv[7].split(",") if node],
    "elapsed_seconds": int(sys.argv[8]),
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

def write_json(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    os.replace(temporary, path)

write_json(health_file, record)
if return_code != 0:
    try:
        with open(scheduler_file, "r") as f:
            scheduler = json.load(f)
    except Exception:
        scheduler = {}
    scheduler.update({
        "state": "failed",
        "phase": "mpi_startup",
        "abort_reason": (
            f"MPI health check failed or timed out with exit code {return_code}; "
            "see mpi_health_check.log"
        ),
        "mpi_health_check": record,
        "updated_at": record["updated_at"],
    })
    write_json(scheduler_file, scheduler)
PY

if [ "$mpi_health_rc" -ne 0 ]; then
    echo "[scheduler] MPI health check failed with exit code $mpi_health_rc on ${allocation_nodes}" >&2
    exit "$mpi_health_rc"
fi
echo "[scheduler] MPI health check passed in ${mpi_health_elapsed}s"
{% endif %}

validate_sizing_checkpoint() {
    python3 {{ scheduler_runner }} --validate-sizing "$sizing_json" "$n_job_tasks"
}

start_time=$(date +%s)
{% if not smooth_only %}
rm -f "$sizing_json"
set +e
if [ "$skip_sizing" = "1" ]; then
    echo "$mpi_exec -n $init_ranks $executable -nthreads $n_threads --job $job_file $fresh_flag --init-no-size"
    "$mpi_exec" "${mpi_args[@]}" -n "$init_ranks" "$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --init-no-size > "$dir_out/init.log" 2>&1
else
    echo "$mpi_exec -n $init_ranks $executable -nthreads $n_threads --job $job_file $fresh_flag --init"
    "$mpi_exec" "${mpi_args[@]}" -n "$init_ranks" "$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --init > "$dir_out/init.log" 2>&1
fi
sizing_rc=$?
set -e
if [ "$sizing_rc" -ne 0 ]; then
    if [ "$skip_sizing" != "1" ] && validate_sizing_checkpoint; then
        echo "[scheduler] sizing exited with $sizing_rc after writing usable estimates; continuing with $sizing_json"
    else
        exit "$sizing_rc"
    fi
fi

python3 {{ scheduler_runner }} \
    --config "$scheduler_config" \
    --job "$job_file" \
    --output "$dir_out" \
    --status "$scheduler_status"
{% else %}
echo "Skipping frequency sweep; running solver postprocess only."
{% endif %}

{% if imaging_job %}
echo "Running solver postprocess step..."
"$mpi_exec" "${mpi_args[@]}" -n "$n_procs" "$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --smooth >> "$dir_out/smooth.log" 2>&1
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
