from dataclasses import dataclass

import numpy as np

from frequensolve.seismic.receivers import ReceiverGroup
from frequensolve.seismic.sources import SourceGroup
from frequensolve.simulation.sampling import Sampling

__all__ = ["ShotRecord"]


@dataclass
class ShotRecord:
    """Container for storing a shot record, including source, receiver info, and field data.

    A ShotRecord may represent frequency-domain (FD) or time-domain (TD) data, along with the
    sampling info. It is aware of the source, receiver configuration, and raw data arrays.

    Attributes:
       type (str): "FD" or "TD" indicating frequency- or time-domain data.
       number (int): Shot number (e.g., source index).
       sampling (Sampling): A Sampling object describing frequency/time ranges.
       source (Source): The associated Source object for this shot.
       receiver_group (ReceiverGroup): The associated ReceiverGroup object.
       field (str): Field name (e.g. "pressure", "displacement").
       data (np.ndarray): The raw data array, shape depends on FD or TD usage.
    """

    type: str
    number: int
    sampling: Sampling
    source: SourceGroup
    receiver_group: ReceiverGroup
    field: str
    data: np.ndarray

    def write_segy(
        self, fname: str, units_in: str = "km", units_out: str = "m", **kwargs
    ):
        """Write a time-domain shot to a SEGY file.

        Uses SEGY to create a valid SEG-Y file with
        correct geometry headers. This method is only valid if the shot type is "TD".

        Args:
           fname (str): Output SEGY file name.
           units_in (str): Units of the input coordinates (defaults to "km").
           units_out (str): Units for the output coordinates (defaults to "m"). Must be 'm' or 'ft'.
           kwargs (dict): Additional options, such as 'Tf' for cutoff time.

        Raises:
           AssertionError: If shot type is not "TD".
           ValueError: If units_out is not "m" or "ft".
        """
        import datetime
        from pathlib import Path

        import pint
        from segy.factory import SegyFactory
        from segy.standards import get_segy_standard

        # Ensure correct shot type
        assert self.type == "TD", "SEGY output is only valid for time-domain (TD) data."

        # Unit conversion checks
        ureg = pint.UnitRegistry()
        if units_out.lower() not in ["m", "ft"]:
            raise ValueError("units_out must be 'm' or 'ft' (meters or feet).")
        iunit = ureg(units_in)
        ounit = ureg.meter if units_out.lower() == "m" else ureg.foot
        scale = iunit.to(ounit).magnitude

        # Basic geometry and dimension info
        group = self.receiver_group
        source = self.source
        dim = len(source.coord)

        # Optional cutoff time
        Tf = kwargs.get("Tf", None)
        nTf, Tf = self.sampling.cutoff(Tf)  # number of time samples after cutoff

        n_traces = group.size
        n_samples = nTf
        interval = int(self.sampling.dT * 1e6)  # sample interval in microseconds
        trace_datetime = datetime.datetime.now()

        # Build SEG-Y config
        config = {
            "spec": get_segy_standard(1.0),
            "samples_per_trace": n_samples,
            "sample_interval": interval,
            "trace_sorting_code": 5,  # Common source point
            "measurement_system_code": "meters" if units_out.lower() == "m" else "feet",
        }

        factory = SegyFactory(**config)
        txt = factory.create_textual_header()
        bin_ = factory.create_binary_header()
        headers = factory.create_trace_header_template(size=n_traces)
        samples = factory.create_trace_sample_template(size=n_traces)

        # Populate headers and data
        for itr in range(n_traces):
            headers[itr]["trace_seq_num_reel"] = itr + 1
            headers[itr]["inline"] = itr + 1
            headers[itr]["crossline"] = 1

            # Source position
            headers[itr]["source_coord_x"] = int(source.coord[0] * scale)
            if dim == 2:
                headers[itr]["source_coord_y"] = 0
            else:
                headers[itr]["source_coord_y"] = int(source.coord[1] * scale)
            headers[itr]["source_surface_elevation"] = int(-source.coord[-1] * scale)

            # Receiver position
            headers[itr]["group_coord_x"] = int(group.coord[itr, 0] * scale)
            if dim == 2:
                headers[itr]["group_coord_y"] = 0
            else:
                headers[itr]["group_coord_y"] = int(group.coord[itr, 1] * scale)
            headers[itr]["receiver_group_elevation"] = int(
                -group.coord[itr, -1] * scale
            )

            # Trace data (slice out the first n_samples from each trace)
            samples[itr] = self.data[:n_samples, itr].copy()

        traces = factory.create_traces(samples=samples, headers=headers)

        # Write the file
        with Path(fname).open(mode="wb") as f:
            f.write(txt)
            f.write(bin_)
            f.write(traces)
