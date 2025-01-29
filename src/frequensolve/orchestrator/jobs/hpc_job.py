"""HPC job class."""

__all__ = ['HPCJob']

class HPCJob:
   """Defines a job to be run on an HPC system."""
   
   def run(self) -> str:
      """Command to run the job."""
      return f"mpirun -np {self.ranks} {self.cmd}"

   # def run_mpi_flux(self, nnodes, ranks_per_node, cmd):
   #    """Submit an MPI job to Flux requesting 'nnodes' nodes."""
   #    h = flux.Flux()  # connect to the local Flux instance

   #    jobspec = flux.job.JobspecV1.from_command(
   #       [cmd],
   #       num_tasks=nnodes * ranks_per_node,
   #       tasks_per_resource_set=ranks_per_node,
   #    )
   #    # jobspec.environment = dict(MYVAR="test")

   #    # Submit the job
   #    jobid = flux.job.submit(h, jobspec)
   #    print(f"Submitted Flux job {jobid} requesting {nnodes} nodes")

   #    # Wait for it to complete and get the return code
   #    event = flux.job.wait(h, jobid)
   #    rc = event["status"]
   #    print(f"Flux job {jobid} completed with status {rc}")

   #    return rc