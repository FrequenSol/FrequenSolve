#!/bin/bash
#
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

{% if job_json %}
input_file={{job_json}}
{% else %}
input_file=$1
{% endif %}
{% if procs_per_task %}
procs_per_task={{procs_per_task}}
{% else %}
procs_per_task=$2
{% endif %}

{% if run_path %}
cd {{run_path}}
{% endif %}

dir_out={{dir_out}}
rm -rf $dir_out
mkdir -p $dir_out
mkdir -p $dir_out

ml intel/25.1 phdf5 petsc/3.23 fftw3
module list

{% if n_tasks > 1 %}
export FS_DISABLE_PARAVIEW=1
{% endif %}
export FS_SOLVER_PATH={{fs_dir}}

export KMP_STACKSIZE=80M

mpi_exec={{mpi}}
executable={{executable}}
n_threads={{n_threads}}
n_procs={{n_procs}}
n_tasks={{n_tasks}}

n_workers=$((n_procs / procs_per_task))

start_time=$(date +%s)

$mpi_exec -n $n_procs $executable $n_threads -j $input_file --init

for i in $(seq 1 $n_tasks); do
   off=$((procs_per_task * ((i-1) % n_workers)))
   echo "$mpi_exec -n $procs_per_task -o $off task_affinity $executable -nthreads $n_threads -j $input_file -i $i"
   $mpi_exec -n $procs_per_task -o $off task_affinity $executable -nthreads $n_threads -j $input_file -i $i >> $dir_out/task_${i}.txt 2>&1 &
   if [[ $((($i - 1) % n_workers)) -eq $((n_workers - 1)) ]]; then
      wait
      echo "Group done"
   fi
done
wait

{% if imaging_job %}
$executable -j $input_file --smooth
{% endif %}

# skip_slots=(1)
# allowed=()
# for ((s=0; s<n_workers; s++)); do
#    skip=0
#    for k in "${skip_slots[@]}"; do
#       if (( s == k )); then skip=1; break; fi
#    done
#    (( skip == 0 )) && allowed+=("$s")
# done

# n_active=3
# if (( n_active == 0 )); then
#    echo "No usable slots after skipping ${skip_slots[*]}"; exit 1
# fi

# launched=0
# for i in $(seq 1 "$n_tasks"); do
#    if ! (( (i - 7) % 16 == 0 )); then
#       continue
#    fi

#    slot=${allowed[$(( launched % n_active ))]}
#    off=$(( procs_per_task * slot ))

#    echo "$mpi_exec -n $procs_per_task -o $off task_affinity $executable -nthreads $n_threads -j $input_file -i $i"
#    $mpi_exec -n "$procs_per_task" -o "$off" task_affinity "$executable" \
#       -nthreads "$n_threads" -j "$input_file" -i "$i" >> "$dir_out/task_${i}.txt" 2>&1 &

#    ((launched++))
#    if (( launched % n_active == 0 )); then
#       wait
#       echo "Group done"
#    fi
# done
# wait

# Calculate and print total time taken
end_time=$(date +%s)
if [ -n "$start_time" ]; then
    total_seconds=$((end_time - start_time))
    hours=$((total_seconds / 3600))
    minutes=$(( (total_seconds % 3600) / 60 ))
    seconds=$((total_seconds % 60))
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
fi
echo "Sweep Complete"
