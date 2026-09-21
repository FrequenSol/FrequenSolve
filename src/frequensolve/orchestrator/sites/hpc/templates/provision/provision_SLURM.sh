#!/bin/bash
#SBATCH -J {{ name }}
#SBATCH -o ./%job.o%j
#SBATCH -e ./%job.e%j
#SBATCH -N {{ nhost }}
#SBATCH -n {{ nproc }}
#SBATCH -p {{ queue }}
{% if account %}
#SBATCH -A {{ account }}
{% endif %}
{% if duration %}
#SBATCH -t {{ duration }}
{% endif %}
{% if notify_email %}
#SBATCH --mail-user={{ notify_email }}
{% endif %}

cd {{ work_dir_shell }}

# {{ mpi }} flux start --boot

# scheduler=dask-scheduler --host 0.0.0.0

# TODO: use dask-mpi to get workers on all ranks
# dask-worker $scheduler --nprocs 1 --nthreads 1

while true; do
   sleep 10
done
