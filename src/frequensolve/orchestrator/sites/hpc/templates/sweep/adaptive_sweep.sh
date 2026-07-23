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
cd {{run_path}}
{% endif %}

dir_out={{dir_out}}
mkdir -p "$dir_out/batch"
find "$dir_out" -mindepth 1 -maxdepth 1 ! -name batch -exec rm -rf -- {} +
scheduler_status="$dir_out/scheduler_status.json"
scheduler_config="$dir_out/scheduler_config.json"
sizing_json={{sizing_json_shell}}
skip_sizing={{skip_sizing}}

{% for line in runtime_setup %}
{{ line }}
{% endfor %}

mpi_exec={{mpi}}
n_procs={{n_procs}}
n_threads={{n_threads}}
export OMP_NUM_THREADS=$n_threads
{% for line in mpi_async_progress_setup %}
{{ line }}
{% endfor %}
n_tasks={{n_tasks}}
n_job_tasks={{n_job_tasks}}
executable={{executable}}
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

validate_sizing_checkpoint() {
    python3 {{ scheduler_runner }} --validate-sizing "$sizing_json" "$n_job_tasks"
}

start_time=$(date +%s)
{% if not smooth_only %}
rm -f "$sizing_json"
set +e
if [ "$skip_sizing" = "1" ]; then
    echo "$mpi_exec -n $n_procs $executable -nthreads $n_threads --job $job_file $fresh_flag --init-no-size"
    $mpi_exec -n $n_procs "$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --init-no-size > "$dir_out/init.log" 2>&1
else
    echo "$mpi_exec -n $n_procs $executable -nthreads $n_threads --job $job_file $fresh_flag --init"
    $mpi_exec -n $n_procs "$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --init > "$dir_out/init.log" 2>&1
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
echo "Skipping frequency sweep; running imaging postprocess only."
{% endif %}

{% if imaging_job %}
echo "Running imaging step..."
$mpi_exec -n "$n_procs" "$executable" -nthreads "$n_threads" --job "$job_file" $fresh_flag --smooth >> "$dir_out/smooth.log" 2>&1
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
