#!/bin/bash
#SBATCH -J {{ name }}
#SBATCH -o ./{{ name }}.o%j
#SBATCH -e ./{{ name }}.e%j
#SBATCH -N {{ nhost }}
#SBATCH -n {{ nproc }}
#SBATCH -p {{ partition }}
{% if account %}
#SBATCH -A {{ account }}
{% endif %}
{% if duration %}
#SBATCH -t {{ duration }}
{% endif %}

cd {{ work_dir }}


{{ mpi }} flux start --boot

scheduler=dask-scheduler --host 0.0.0.0

# TODO: use dask-mpi to get workers on all ranks
dask-worker $scheduler --nprocs 1 --nthreads 1

while true; do
   sleep 10
done