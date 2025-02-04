import os
import sys

from frequensolve import *
from frequensolve.seismic import *

if __name__ == "__main__":
    input_file = sys.argv[1]

    acq = Acquisition.from_file(input_file, upscale=2)

    A = 0.1  # Amplitude (for plotting)

    # Plot wavelet
    acq.source_group.signature(1).plot()

    for field in acq.list_fields():
        for source in acq.list_sources():
            print(field, source)

            if "field" in field:
                shot_record = acq.read_shot_TD(field, source)
                animate_gather(
                    shot_record,
                    A=A,
                    Tf=40,
                    cmap="RdGy",
                    units="km",
                    figsize=(8, 4),
                    interval=10,
                    save="movie",
                )
                del shot_record
            else:
                shot_record = acq.read_shot_TD(field, source)
                plot_gather(shot_record, A=A, Tf=3.2, cmap="binary_r", units="km")
                del shot_record
