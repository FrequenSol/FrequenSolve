# FrequenSolve on Stampede3: collaborator setup and workflow

This guide is for collaborators who will run FrequenSolve Python scripts or
notebooks on a laptop or workstation and submit the actual solver work to
Stampede3. It assumes only basic familiarity with a terminal.

FrequenSolve has two parts:

- The **Python package** runs on your local computer. It builds projects, stages
  files, submits SLURM jobs, monitors them, and retrieves results.
- The licensed **`FS_seismic` solver** is already installed on Stampede3. Do
  not copy it to your computer and do not run it on a Stampede3 login node.
  FrequenSolve submits it to SLURM compute nodes for you.

The shared solver executable used in this guide is:

```text
/work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic
```

> [!IMPORTANT]
> As of July 20, 2026, the newest PyPI release is `frequensolve==0.2.1`.
> PyPI 0.2.1 uses an older Stampede3 configuration format and cannot put a
> custom solver path in `site.toml`. The current `dev2` source version supports
> the complete `site.toml` configuration requested here. **Use the GitHub
> `dev2` installation for current Stampede3 work unless your group confirms
> that a newer compatible PyPI release has been published.** A working PyPI
> 0.2.1 compatibility setup is included later in this guide.

## 1. Before you begin

You need:

- A TACC account, MFA pairing, and access to an active Stampede3 allocation.
- Your TACC username and, if required, the allocation/project name to charge.
- A local Mac or Linux computer with Python 3.10 through 3.14, `ssh`, and `git`.
- `rsync` if possible. FrequenSolve can use SFTP if `rsync` is unavailable.

Open a terminal **on your local computer**, then check the basic programs:

```bash
python3 --version
ssh -V
git --version
rsync --version
```

Commands in this guide marked as local must be run on your computer, not after
logging in to Stampede3.

## 2. Install the FrequenSolve Python package

Choose **one** installation method. A virtual environment keeps FrequenSolve
and its dependencies separate from the rest of your computer.

### Option A: install a released version from PyPI

Run on your local computer:

```bash
mkdir -p "$HOME/.venvs"
python3 -m venv "$HOME/.venvs/frequensolve-pypi"
source "$HOME/.venvs/frequensolve-pypi/bin/activate"
python -m pip install --upgrade pip
python -m pip install "frequensolve[hpc,visual]" jupyterlab ipykernel
python -c 'import frequensolve as fs; print("FrequenSolve", fs.__version__)'
```

The prompt normally gains a prefix such as `(frequensolve-pypi)` while the
environment is active. In every new terminal, reactivate it with:

```bash
source "$HOME/.venvs/frequensolve-pypi/bin/activate"
```

If the reported version is `0.2.1`, use the **PyPI 0.2.1 compatibility
configuration** in section 5. Do not use the current `dev2` TOML file with
that release.

PyPI installs the library but does not provide an editable source checkout or
the repository's tutorial notebooks. Use your group's scripts/notebooks, or
clone the repository separately to obtain examples.

### Option B: clone GitHub `dev2` and install from source (recommended now)

Choose a parent directory for source code, such as `$HOME/src`, and run on your
local computer:

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone https://github.com/FrequenSol/FrequenSolve.git
cd FrequenSolve
git switch dev2
git pull --ff-only origin dev2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hpc,visual]" jupyterlab ipykernel
python -c 'import frequensolve as fs; print("FrequenSolve", fs.__version__)'
```

The `-e` installation is editable: Python uses the code in this checkout, so a
later `git pull` updates the installed source without copying it elsewhere.
Reactivate this environment in every new terminal with:

```bash
cd "$HOME/src/FrequenSolve"
source .venv/bin/activate
```

To update the source installation later:

```bash
cd "$HOME/src/FrequenSolve"
git switch dev2
git pull --ff-only origin dev2
source .venv/bin/activate
python -m pip install -e ".[hpc,visual]"
```

## 3. Configure SSH and connection sharing

Stampede3 requires MFA. FrequenSolve can reuse an authenticated OpenSSH master
connection, which prevents a new password/MFA prompt for every transfer or
remote command.

### Optional: create a local SSH key

A local key is optional for the password-plus-MFA workflow below. Generate one
only if you do not already have one and your institution or TACC support has
instructed you how to register its **public** half:

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
ssh-keygen -t ed25519 -a 100 -f "$HOME/.ssh/id_ed25519" -C "YOUR_EMAIL_ADDRESS"
```

Choose a passphrase when prompted. The private file is
`~/.ssh/id_ed25519`; never share it. The public file is
`~/.ssh/id_ed25519.pub`.

> [!WARNING]
> Run `ssh-keygen` only on your local computer. TACC explicitly warns users not
> to run it on Stampede3 because changing the remote `~/.ssh` setup can
> interfere with batch jobs. A client key does not remove the MFA requirement.

### Configure a master control socket

Create the socket directory and protect it:

```bash
mkdir -p "$HOME/.ssh/control"
chmod 700 "$HOME/.ssh" "$HOME/.ssh/control"
touch "$HOME/.ssh/config"
chmod 600 "$HOME/.ssh/config"
nano "$HOME/.ssh/config"
```

Add the following. Replace `YOUR_TACC_USERNAME`. If the file already contains a
`Host *` block, add the four shared options to that existing block instead of
creating conflicting copies.

```sshconfig
Host stampede3
  HostName stampede3.tacc.utexas.edu
  User YOUR_TACC_USERNAME

Host *
  ServerAliveInterval 30
  ControlPersist 10m
  ControlPath ~/.ssh/control/%C
  ControlMaster auto
```

The value is literally `%C`; do not include Markdown `**` characters. In
`nano`, save with <kbd>Ctrl</kbd>+<kbd>O</kbd>, press <kbd>Enter</kbd>, then
exit with <kbd>Ctrl</kbd>+<kbd>X</kbd>.

`ControlPersist 10m` keeps the authenticated socket alive for ten minutes after
the last SSH client exits. Keeping an interactive login open in another
terminal is even more reliable during a notebook run.

### Make the first connection before using Python

In terminal 1 on your local computer, run:

```bash
ssh stampede3
```

On the first connection, verify the host-key fingerprint through TACC before
accepting it. Enter your TACC password and MFA code when prompted. Once logged
in, verify your allocation, current partitions, and the shared solver:

```bash
hostname
qlimits
sinfo -S+P -o "%18P %8a %20F"
test -x /work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic \
  && echo "FS_seismic is accessible and executable"
```

If the final message does not appear, stop and ask the FrequenSolve
administrator to grant access or confirm the path.

Leave terminal 1 logged in while starting FrequenSolve from terminal 2. Do not
run the solver manually on the login node.

## 4. Create `~/.frequensolve/site.toml` for `dev2`

This section applies to the GitHub `dev2` installation and to a future PyPI
release that supports `type = "slurm"`, `preset = "stampede3"`, and a
`solver` field.

The file is on your **local computer**, even though the paths inside the
Stampede3 profiles refer to the remote system. FrequenSolve can create a
starter file automatically on the first `fs.Site()` call, but the following is
simpler when using this complete configuration:

```bash
mkdir -p "$HOME/.frequensolve"
chmod 700 "$HOME/.frequensolve"
nano "$HOME/.frequensolve/site.toml"
```

Paste the complete configuration below. Replace every
`YOUR_TACC_USERNAME`. If TACC requires an explicit allocation, uncomment and
replace each `account` line. Otherwise leave the `account` lines commented.

```toml
# FrequenSolve execution profiles on this computer.
# fs.Site() uses the profile named by `default`.
default = "stampede3"

# Default Stampede3 profile: conservative settings for initial checks.
[sites.stampede3]
type = "slurm"
preset = "stampede3"
username = "YOUR_TACC_USERNAME"
credential = "tacc-stampede3"
solver = "/work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic"
default_partition = "skx-dev"
transfer_method = "rsync"
modules = []
verbose = true
# account = "YOUR_TACC_ALLOCATION"
# ssh_key = "~/.ssh/id_ed25519"  # Only if this key is registered and usable.

[sites.stampede3.run_config]
nodes = 1
duration = "00:30:00"
ranks_per_node = 2
ranks_per_task = 1
poll_interval = 10

# Explicit debug profile using the current Stampede3 SKX development partition.
[sites.stampede3-debug]
type = "slurm"
preset = "stampede3"
username = "YOUR_TACC_USERNAME"
credential = "tacc-stampede3"
solver = "/work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic"
default_partition = "skx-dev"
transfer_method = "rsync"
modules = []
verbose = true
# account = "YOUR_TACC_ALLOCATION"
# ssh_key = "~/.ssh/id_ed25519"

[sites.stampede3-debug.run_config]
nodes = 1
duration = "00:30:00"
ranks_per_node = 2
ranks_per_task = 1
poll_interval = 10

# Production profile: 4 SPR nodes and 8 MPI ranks per node.
[sites.stampede3-prod]
type = "slurm"
preset = "stampede3"
username = "YOUR_TACC_USERNAME"
credential = "tacc-stampede3"
solver = "/work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic"
default_partition = "spr"
transfer_method = "rsync"
modules = []
verbose = true
# account = "YOUR_TACC_ALLOCATION"
# ssh_key = "~/.ssh/id_ed25519"

[sites.stampede3-prod.run_config]
nodes = 4
duration = "02:00:00"
ranks_per_node = 8
ranks_per_task = 1
poll_interval = 10
```

The three profile tables are intentionally complete. Current `site.toml` does
not provide profile inheritance, so connection details must be repeated.

The top line configures the default profile used by `fs.Site()`. The default
profile is named `stampede3` and uses the same conservative settings as the
debug profile. To make production the default later, change only:

```toml
default = "stampede3-prod"
```

If local `rsync` is unavailable, change `transfer_method = "rsync"` to
`transfer_method = "sftp"` in each profile.

The default remote work base is `$WORK/frequensolve`; FrequenSolve asks the
remote login shell for `$WORK`. You do not need to create it or put a literal
`$WORK` value in TOML. The local `~/.frequensolve` directory is unrelated to
that remote work directory.

## 5. PyPI 0.2.1 compatibility configuration

Skip this section when using GitHub `dev2`.

PyPI 0.2.1 uses the legacy `type = "stampede3"` profile, the names
`procs_per_node` and `procs_per_task`, and environment variables for the TACC
username and solver. It cannot store the solver path in `site.toml`.

Before starting Python, a script, or Jupyter in each new local terminal, run:

```bash
source "$HOME/.venvs/frequensolve-pypi/bin/activate"
export TACC_USERNAME="YOUR_TACC_USERNAME"
export STAMPEDE3_SOLVER_EXECUTABLE="/work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic"
```

Then create `~/.frequensolve/site.toml` as described above, but use this legacy
content:

```toml
default = "stampede3"

[sites.stampede3]
type = "stampede3"
rel_path = "frequensolve"
queue = "skx-dev"
nodes = 1
duration = "00:30:00"
procs_per_node = 2
procs_per_task = 1
poll_interval = 10
verbose = true
# account = "YOUR_TACC_ALLOCATION"

[sites.stampede3-debug]
type = "stampede3"
rel_path = "frequensolve"
queue = "skx-dev"
nodes = 1
duration = "00:30:00"
procs_per_node = 2
procs_per_task = 1
poll_interval = 10
verbose = true
# account = "YOUR_TACC_ALLOCATION"

[sites.stampede3-prod]
type = "stampede3"
rel_path = "frequensolve"
queue = "spr"
nodes = 4
duration = "02:00:00"
procs_per_node = 8
procs_per_task = 1
poll_interval = 10
verbose = true
# account = "YOUR_TACC_ALLOCATION"
```

The master SSH socket still avoids repeated MFA. Upgrade to the current
configuration in section 4 when a compatible release reaches PyPI.

## 6. Test the configuration before submitting a job

Keep terminal 1 connected to Stampede3. In terminal 2, activate the environment
that contains FrequenSolve, then run the appropriate test.

For the GitHub source installation:

```bash
cd "$HOME/src/FrequenSolve"
source .venv/bin/activate
python - <<'PY'
import frequensolve as fs

site = fs.Site()
print("backend:", type(site).__name__)
print("login host:", site.login_host)
print("remote work directory:", site.work_dir)
print("solver executable:", site.executable)
site.close()
PY
```

For PyPI 0.2.1, activate its environment and export the two variables from
section 5 before running the same test.

Expected output includes:

- Backend `SlurmSite` on `dev2`, or `Stampede3Site` on PyPI 0.2.1.
- Login host `stampede3.tacc.utexas.edu`.
- A remote work path below your Stampede3 `$WORK` directory.
- The `FS_v1.3.0/FS_seismic` executable path.

Site creation opens an SSH connection, so an authentication or path error at
this step should be fixed before trying a real simulation.

## 7. Normal working procedure

Use this order each time you work.

### Terminal 1: authenticate to Stampede3 first

```bash
ssh stampede3
```

Complete password/MFA authentication and leave this terminal connected. This
creates the control socket that FrequenSolve will reuse.

### Terminal 2: activate FrequenSolve

For the source installation:

```bash
cd "$HOME/src/FrequenSolve"
source .venv/bin/activate
```

For PyPI 0.2.1:

```bash
source "$HOME/.venvs/frequensolve-pypi/bin/activate"
export TACC_USERNAME="YOUR_TACC_USERNAME"
export STAMPEDE3_SOLVER_EXECUTABLE="/work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic"
```

### Run a Python script

```bash
python path/to/your_script.py
```

A typical submission section in a script is:

```python
import frequensolve as fs

# Build a job in the script, or load a previously saved job.
job = fs.SimulationJob.load("/path/to/job.json")

site = fs.Site()  # Uses the profile named by `default` in site.toml.
result = site.submit(job).wait()

print("status:", result.status)
print("successful:", result.successful)
print("logs:", result.logs())
site.close()
```

### Run a notebook

Start Jupyter from the activated environment:

```bash
python -m jupyter lab
```

Open the notebook in the browser. If Jupyter asks for a kernel, select the
Python kernel from the activated FrequenSolve environment. Source users can
start with `examples/tutorials/02_sites/02_hpc_sites.ipynb` after checking its
project and job settings.

The notebook should create the site with `fs.Site()` or select a named profile
with `fs.Site(profile="stampede3-debug")`. Do not start Jupyter on a Stampede3
login node for this workflow; it runs locally and submits the compute work.

## 8. Select profiles and override resources at runtime

The TOML values are defaults, not fixed limits. `fs.Site()` uses the profile
named by the top-level `default` setting:

```python
site = fs.Site()  # [sites.stampede3] with the supplied file
```

Select either named profile without editing the file:

```python
debug_site = fs.Site(profile="stampede3-debug")
prod_site = fs.Site(profile="stampede3-prod")
```

With `dev2`, override a profile while constructing the site:

```python
site = fs.Site(
    profile="stampede3-prod",
    queue="spr",
    nodes=2,
    ranks_per_node=8,
    duration="01:00:00",
)
result = site.submit(job).wait()
```

Or override only one submission made through an existing site:

```python
site = fs.Site(profile="stampede3-prod")
result = site.submit(
    job,
    queue="spr",
    nodes=8,
    ranks_per_node=8,
    duration="04:00:00",
).wait()
```

For PyPI 0.2.1, use `procs_per_node` instead of `ranks_per_node`.

Check `qlimits` before increasing nodes or duration. The request must fit both
the live Stampede3 queue limits and your allocation.

## 9. Recommended MPI rank layout

Use these FrequenSolve starting points unless a benchmark for your model shows
that another layout is better:

| Stampede3 node type | Partition(s) | Recommended ranks per node |
| --- | --- | ---: |
| Sapphire Rapids (SPR) | `spr` | **8**, especially for large multi-node runs |
| Skylake (SKX) | `skx`, `skx-dev` | **2** |
| Ice Lake (ICX) | `icx` | **2** |

FrequenSolve uses the remaining cores as threads within each MPI rank. For
example, the production default requests 4 SPR nodes × 8 ranks per node = 32
MPI ranks in total.

The profile defaults are only convenient starting values. Node count, rank
layout, duration, and partition may all be changed at runtime as shown above.

> [!NOTE]
> `stampede3-debug` is the FrequenSolve profile name; `skx-dev` is the
> Stampede3 partition selected by that profile.

## 10. Common problems

### Repeated password or MFA prompts

Confirm that terminal 1 is still connected and that a socket exists:

```bash
ls -la "$HOME/.ssh/control"
ssh -O check stampede3
```

If the socket expired, reconnect with `ssh stampede3` and complete MFA again.

### `Unknown SLURM partition` or a rejected resource request

Use `skx-dev` for the current SKX development partition. Run `qlimits` on
Stampede3 because TACC can change queue policy after this guide is published.

### Solver missing or not executable

The `solver` value must be the complete executable path ending in
`FS_seismic`, not only the `FS_v1.3.0/` directory. Verify it from an SSH login:

```bash
ls -l /work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic
test -x /work2/06472/jbadger/shared/stampede3/FS_v1.3.0/FS_seismic
```

### PyPI reports an unexpected TOML keyword

Check the installed version:

```bash
python -c 'import frequensolve as fs; print(fs.__version__)'
```

Version 0.2.1 needs the legacy section 5 configuration. Use GitHub `dev2` for
the current `solver`-in-TOML setup.

### Python cannot import FrequenSolve, or Jupyter uses the wrong package

Reactivate the correct environment, then check both executables:

```bash
which python
python -c 'import frequensolve as fs; print(fs.__version__, fs.__file__)'
```

Start Jupyter with `python -m jupyter lab` from that same terminal.

### Transfer fails because `rsync` is missing

Install local `rsync`, or set `transfer_method = "sftp"` in every profile.

### Stampede3 login succeeds but job submission reports an account error

Uncomment `account = "YOUR_TACC_ALLOCATION"` in every profile and replace it
with the exact allocation name shown by TACC. If you have only one default
allocation, leaving it unset is usually preferable.

## References

- [FrequenSolve on PyPI](https://pypi.org/project/frequensolve/)
- [FrequenSolve GitHub repository](https://github.com/FrequenSol/FrequenSolve)
- [Stampede3 user guide](https://docs.tacc.utexas.edu/hpc/stampede3/)
- [TACC multi-factor authentication](https://docs.tacc.utexas.edu/basics/mfa/)
