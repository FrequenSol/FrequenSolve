"""Generated Cloud benchmark workload.

Source tutorial: 00_getting_started/acoustic_starter.ipynb
Source SHA-256: cc833850e77a81786fe118991ea11a5e02272620371e6d0fca58a8bcf4bcda1d
"""

# %% source cell 1
from pathlib import Path

from acoustic_starter import prepare, submit

workspace = Path.cwd() / "acoustic-starter-v1"
project, job = prepare(workspace)
print(project.path)
print(job.f_list)

# %% source cell 3
result = submit(workspace, profile="cloud")

# %% source cell 5
traces = result.traces()
print(traces.summary)
group = traces.groups[0]
component = traces.components(group)[0]
source = traces.sources(group)[0]
response = traces.fd(group, component, source)
amplitude = abs(response).isel(frequency=0)
axis.set_xlabel("Receiver")
units = response.attrs.get("units", "units not reported")
axis.set_ylabel(f"Pressure magnitude ({units})")
axis.set_title("Acoustic starter: 10 Hz receiver response")
