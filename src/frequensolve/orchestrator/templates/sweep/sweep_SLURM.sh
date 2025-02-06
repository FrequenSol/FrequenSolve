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

ntask={{ntask}}
nhost=$((nrank / ranks_per_job))

rm -rf ./jobs/out/
mkdir -p ./jobs/out/

for i in $(seq 1 $ntask); do
   off=$((tasks_per_job * ((i-1) % nhost)))
   echo "{{ mpi }} -n $tasks_per_job -o $off ./FS_seismic -nthreads {{ threads }} -j $input_file -i $i "
   {{ mpi }} -n $tasks_per_job -o $off task_affinity {{ executable }} -nthreads {{ threads }} -j $1 -i $i >> ./jobs/out/j${i}.txt 2>&1 &
   if [[ $((($i - 1) % nhost)) -eq $((nhost - 1)) ]]; then
      wait
      echo "Group done"
   fi
done
wait
