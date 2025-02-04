#!/bin/bash
#PBS -N {{ name }}
#PBS -o ./${PBS_JOBID}.out
#PBS -e ./${PBS_JOBID}.err
#PBS -l nodes={{ nhost }}:ppn={{ nproc }}
{% if duration %}
#PBS -l walltime={{ duration }}
{% endif %}

cd {{ work_dir }}

{{ mpi }} flux start --boot
scheduler=dask-scheduler --host 0.0.0.0

# TODO: use dask-mpi to get workers on all ranks
dask-worker $scheduler --nprocs 1 --nthreads 1

while true; do
   sleep 10
done
