"""
slurm_executor.py

Implements an executor that uses SLURM to submit jobs. 
This is a simplistic demonstration using subprocess calls to `sbatch` or `srun`.
In practice, you might want more advanced scheduling options, job status checks, etc.
"""

import subprocess
import tempfile
import textwrap
import time
from typing import Optional
from executor_base import ExecutorBase, ExecutionResult


class SlurmExecutor(ExecutorBase):
    """
    SLURM executor that creates a temporary Slurm script, submits it, and optionally 
    waits for the job to complete. Real implementations often require job tracking,
    logging, etc.
    """

    def __init__(
        self,
        partition: str = "debug",
        time: str = "00:30:00",
        account: Optional[str] = None,
        wait: bool = True,
    ):
        self.partition = partition
        self.time = time
        self.account = account
        self.wait = wait

    def run_command(self, command: str, job_name="my_job", *args, **kwargs) -> ExecutionResult:
        """
        Submits the command as a Slurm job using sbatch. If wait=True, we block 
        until the job finishes, returning the final output.
        """
        slurm_script_content = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH --partition={self.partition}
            #SBATCH --time={self.time}
            """)
        if self.account:
            slurm_script_content += f"#SBATCH --account={self.account}\n"

        # Additional SLURM directives...
        slurm_script_content += f"\n{command}\n"

        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".slurm") as tmp:
            tmp.write(slurm_script_content)
            tmp.flush()
            script_path = tmp.name

        # Submit job
        submit_cmd = f"sbatch {script_path}"
        submit_proc = subprocess.run(
            submit_cmd, shell=True, capture_output=True, text=True
        )

        if submit_proc.returncode != 0:
            return ExecutionResult(
                stdout=submit_proc.stdout,
                stderr=submit_proc.stderr,
                return_code=submit_proc.returncode
            )

        # Extract job ID from output, e.g., "Submitted batch job 123456"
        stdout_str = submit_proc.stdout.strip()
        job_id = None
        for word in stdout_str.split():
            if word.isdigit():
                job_id = word
                break

        if not self.wait:
            # We don't wait for job completion, just return
            return ExecutionResult(
                stdout=stdout_str, stderr="", return_code=0, job_id=job_id
            )

        # Wait loop: poll job status until it completes
        while True:
            squeue_cmd = f"squeue -j {job_id}"
            squeue_proc = subprocess.run(
                squeue_cmd, shell=True, capture_output=True, text=True
            )
            if job_id not in squeue_proc.stdout:
                # Job is no longer in squeue, assume finished
                break
            time.sleep(5)  # Sleep a bit, then check again

        # Once finished, we don't necessarily have direct access to stdout/stderr unless
        # the user has them written to SLURM's standard output/error files. 
        # For demonstration, we return a success code here.
        return ExecutionResult(
            stdout=f"Job {job_id} completed.",
            stderr="",
            return_code=0,
            job_id=job_id
        )

    def run_mpi_command(self, command: str, nproc: int, job_name="mpi_job", *args, **kwargs) -> ExecutionResult:
        """
        We can utilize srun within the job script, or rely on mpirun. 
        Here we show a simple example with srun. 
        """
        mpi_cmd = f"srun -n {nproc} {command}"
        return self.run_command(mpi_cmd, job_name=job_name)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        For HPC, you might need to do an scp or rely on a shared filesystem.
        If the HPC cluster shares a filesystem with the submit node, 
        you could just copy locally. In a real scenario, you'd do e.g.:
            subprocess.run(["scp", local_path, f"{remote_user}@{remote_host}:{remote_path}"])
        """
        # This example assumes shared filesystem between login node and compute nodes:
        import shutil
        shutil.copyfile(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Opposite of upload_file.
        """
        import shutil
        shutil.copyfile(remote_path, local_path)
        
   def run_array_command(
      self,
      command: str,
      array_range: str = "1-10",
      job_name: str = "my_array_job",
      *args,
      **kwargs
   ) -> ExecutionResult:
      # Similar to run_command, but we add an --array directive:
      slurm_script_content = textwrap.dedent(f"""\
         #!/bin/bash
         #SBATCH --job-name={job_name}
         #SBATCH --array={array_range}
         #SBATCH --partition={self.partition}
         #SBATCH --time={self.time}
      """)
      if self.account:
         slurm_script_content += f"#SBATCH --account={self.account}\n"
      slurm_script_content += f"\n{command}\n"

    # Submit via sbatch; parse job ID; optionally wait for completion, etc.






#import os
#import subprocess
#from mpi4py import MPI
#from concurrent.futures import ThreadPoolExecutor
#
#def launch_task(command):
#    """Launches a single task using subprocess."""
#    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#    return result.stdout.decode('utf-8'), result.stderr.decode('utf-8')
#
#def distribute_jobs(input_file, tasks_per_job, nhosts, total_tasks):
#
#    """Distribute jobs across available hosts."""
#    
#    n_jobs = int(subprocess.check_output(f"python3 {os.getenv('FREQUENSOL_DIR')}/scripts/read_input.py {input_file}", shell=True).strip())
#    print(f"Total jobs to run: {n_jobs}")
#    
#    job_commands = []
#    for j in range(1, n_jobs + 1):
#        offset = tasks_per_job * ((j - 1) % nhosts)
#        command = f"ibrun -n {tasks_per_job} -o {offset} ./FS_seismic -nthreads 28 -i {input_file} -j {j}"
#        job_commands.append(command)
#    
#    with ThreadPoolExecutor(max_workers=nhosts) as executor:
#        futures = []
#        for idx, command in enumerate(job_commands):
#            futures.append(executor.submit(launch_task, command))
#            if (idx + 1) % nhosts == 0:
#                # Wait for current batch to finish
#                for future in futures:
#                    stdout, stderr = future.result()
#                    print(stdout)
#                    if stderr:
#                        print(f"Error: {stderr}")
#                futures.clear()
#                print("Group done")
#        
#        # Wait for remaining tasks
#        for future in futures:
#            stdout, stderr = future.result()
#            print(stdout)
#            if stderr:
#                print(f"Error: {stderr}")
#
#if __name__ == "__main__":
#    input_file = os.environ.get("INPUT_FILE", "input.dat")
#    tasks_per_job = int(os.environ.get("TASKS_PER_JOB", 2))
#    nhosts = int(os.environ.get("NHOSTS", 10))  # Modify as needed
#    
#    # Assuming SLURM environment variables
#    nodes = int(os.environ.get("SLURM_NNODES", 1))
#    total_tasks = int(os.environ.get("SLURM_NTASKS", 2))
#    
#    distribute_jobs(input_file, tasks_per_job, nhosts, total_tasks)
