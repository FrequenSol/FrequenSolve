#!/bin/bash
#
{% if batch_job %}
#SBATCH -J {{ name }}
#SBATCH -o ./{{ name }}.o%j
#SBATCH -e ./{{ name }}.e%j
#SBATCH -N {{ nnode }}
#SBATCH -n {{ nrank }}
#SBATCH -p {{ queue }}
{% if account %}
#SBATCH -A {{ account }}
{% endif %}
{% if duration %}
#SBATCH -t {{ duration }}
{% endif %}
{% endif %}

input_file=$1
ranks_per_job=$2

export FREQUENSOL_SWEEP=1
export FREQUENSOLVE_DIR={{fs_dir}}
export FS_PROJECT_DIR={{project_dir}}

nranks={{nrank}}
nfreq={{njob}}
nnodes=$((nranks / ranks_per_job))

rm -rf ./jobs/out/
mkdir -p ./jobs/out/

for i in $(seq 1 $nfreq); do
   off=$((ranks_per_job * ((i-1) % nnodes)))
   echo "{{ mpi }} -n $ranks_per_job -o $off {{ executable }} -nthreads {{ nthread }} -j $input_file -i $i "
   {{ mpi }} -n $ranks_per_job -o $off task_affinity {{ executable }} -nthreads {{ nthread }} -j $1 -i $i >> ./jobs/out/j${i}.txt 2>&1 &
   if [[ $((($i - 1) % nnodes)) -eq $((nnodes - 1)) ]]; then
      wait
      echo "Group done"
   fi
done

echo "Sweep Complete"
