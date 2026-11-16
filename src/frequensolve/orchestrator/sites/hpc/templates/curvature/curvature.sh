#!/bin/bash
{% if batch_job %}
#SBATCH -J {{ name }}
#SBATCH -o {{ launch_log }}
#SBATCH -e {{ launch_log }}
#SBATCH -N {{ n_nodes }}
#SBATCH -n {{ n_procs }}
{% if cpus_per_task %}
#SBATCH --ntasks-per-node={{ ranks_per_node }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
{% endif %}
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

status_file={{ status_shell }}
record_exit_status() {
    status=$?
    printf '%s\n' "$status" > "$status_file.partial" && mv -f "$status_file.partial" "$status_file"
}
trap record_exit_status EXIT
printf '%s\n' "$$" > {{ pid_shell }}
cd {{ directory_shell }}

{% for line in runtime_setup %}
{{ line }}
{% endfor %}

mpi_exec={{ mpi_shell }}
mpi_args=( {{ mpi_args_shell }} )
executable={{ executable_shell }}
request={{ request_shell }}
ranks={{ n_procs }}
threads={{ n_threads }}
export OMP_NUM_THREADS=$threads
# Give every rank its own cores, as the sweep scheduler does per launcher.
bind_threads=1
case "${mpi_exec##*/}" in
    srun) launch=(--ntasks "$ranks" --cpus-per-task "$threads") ;;
    ibrun) launch=(-n "$ranks" -o 0 task_affinity) ;;
    *)
        launch=(-n "$ranks")
        version=$("$mpi_exec" --version 2>&1 || true)
        if [ "$(uname -s)" = Linux ] && [[ "$version" == *"Open MPI"* ]]; then
            launch+=(--map-by "slot:PE=$threads" --bind-to core)
        else
            # Unbound ranks would share pinned thread places; leave both free.
            export PRTE_MCA_hwloc_default_binding_policy="${PRTE_MCA_hwloc_default_binding_policy:-none}"
            export OMPI_MCA_hwloc_base_binding_policy="${OMPI_MCA_hwloc_base_binding_policy:-none}"
            bind_threads=0
        fi ;;
esac
{% if thread_placement %}
if [ "$bind_threads" = 1 ]; then
{% for line in thread_placement %}
    {{ line }}
{% endfor %}
fi
{% endif %}
echo "$mpi_exec ${launch[*]} $executable -nthreads $threads --curvature $request"
"$mpi_exec" ${mpi_args[@]+"${mpi_args[@]}"} "${launch[@]}" "$executable" \
    -nthreads "$threads" --curvature "$request" > solver.log 2>&1
