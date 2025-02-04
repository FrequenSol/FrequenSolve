#!/bin/bash
#
#SBATCH -J sweep
#SBATCH -o ./sweep.o%j
#SBATCH -e ./sweep.e%j
#SBATCH -p development
#SBATCH -N 40
#SBATCH -n 80
#SBATCH -t 02:00:00
#SBATCH -A EAR24002

# Usage
#  When called directly:
#     ./sweep.slurm <Input_file> <tasks_per_job> <# of Nodes>
#
#  When called via SLURM:
#     ./sweep.slurm <Input_file> <tasks_per_job>

input_file=$1
tasks_per_job=$2

if [ ! -z "$SLURM_NNODES" ]; then
   if [ $# -eq 2 ]; then
      module list
      pwd
      date

      nodes=$SLURM_NNODES
      tasks=$SLURM_NTASKS
   else
      echo "Usage: sbatch sweep.slurm <Input_file> <tasks_per_job>"
      exit 1
   fi
else
   if [ $# -eq 3 ]; then
      nodes=$3
      tasks=$((2 * nodes))
   else
      echo "Usage: sweep.slurm <Input_file> <tasks_per_job> <# of Nodes>"
      exit 1
   fi
fi

nhosts=$((tasks / tasks_per_job))
tasks_per_node=$((tasks / nodes))
threads=$((56 / tasks_per_node))

# Let code know we're doing a parallel sweep (disable paraview output, etc.)
export FREQUENSOL_SWEEP=1

n_job=$(python3 ${FREQUENSOL_DIR}/scripts/FS_pytools/input_parser.py "$input_file")

rm -rf ./jobs/
mkdir -p ./jobs

for j in $(seq 1 $n_job); do
   off=$((tasks_per_job * ((j-1) % nhosts)))
   echo "ibrun -n $tasks_per_job -o $off ./FS_seismic -nthreads $threads -i $input_file -j $j "
   ibrun -n $tasks_per_job -o $off task_affinity ./FS_seismic -nthreads 28 -i $1 -j $j >> ./jobs/job${j}.txt 2>&1 &
   if [[ $((($j - 1) % nhosts)) -eq $((nhosts - 1)) ]]; then
      wait
      echo "Group done"
   fi
done
wait
