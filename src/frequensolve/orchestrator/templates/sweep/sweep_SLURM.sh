#!/bin/bash
#
{% if batch_job %}
#SBATCH -J {{ name }}
{% if run_path %}
#SBATCH -o {{run_path}}/jobs/batch/job.o%j
#SBATCH -e {{run_path}}/jobs/batch/job.e%j
{% else %}
#SBATCH -o ./job.o%j
#SBATCH -e ./job.e%j
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
{% if batch_job %}
dir_out=jobs/batch/$SLURM_JOB_ID
mkdir -p $dir_out
{% else %}
dir_out=jobs/out/
rm -rf $dir_out
mkdir -p $dir_out
{% endif %}

{% if n_tasks > 1 %}
export FREQUENSOL_SWEEP=1              # Disable ParaView output
{% endif %}
export FREQUENSOLVE_DIR={{fs_dir}}

mpi_exec={{mpi}}
executable={{executable}}
n_threads={{n_threads}}
n_procs={{n_procs}}
n_tasks={{n_tasks}}

n_workers=$((n_procs / procs_per_task))

for i in $(seq 1 $n_tasks); do
   off=$((procs_per_task * ((i-1) % n_workers)))
   echo "$mpi_exec -n $procs_per_task -o $off task_affinity $executable -nthreads $n_threads -j $input_file -i $i"
   $mpi_exec -n $procs_per_task -o $off task_affinity $executable -nthreads $n_threads -j $input_file -i $i >> $dir_out/j${i}.txt 2>&1 &
   if [[ $((($i - 1) % n_workers)) -eq $((n_workers - 1)) ]]; then
      wait
      echo "Group done"
   fi
done
wait

echo "Sweep Complete"
