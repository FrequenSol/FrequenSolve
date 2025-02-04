#BSUB -J {{ name }}
#BSUB -o ./%J.out
#BSUB -e ./%J.err
#BSUB -n {{ nproc }}
#BSUB -R "span[hosts=1]"
{% if duration %}
#BSUB -W {{ duration }}
{% endif %}

cd {{ work_dir }}

{{ mpi }} flux start --boot
scheduler=dask-scheduler --host 0.0.0.0

# TODO: use dask-mpi to get workers on all ranks
dask-worker $scheduler --nprocs 1 --nthreads 1

while true; do
   sleep 10
done
