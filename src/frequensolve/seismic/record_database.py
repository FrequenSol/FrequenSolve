"""
Not used yet; this will complement survey.py for reading and visualizing
data when finished.

Right now this is just a hodge-podge of code that was displaced in
the refactoring process.
"""

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import segyio

from frequensolve.seismic.shot_record import ShotRecord
from frequensolve.simulation.sampling import UniformSweepSampling
from frequensolve.simulation.simulation import SeismicSimulation


@dataclass
class RecordDatabase:
    """Database for managing seismic shot records."""

    sim: SeismicSimulation

    def load_shot_FD(self, key: str, isrc: int) -> ShotRecord:
        """Read frequency-domain shot data, then apply the wavelet signature.

        Args:
           key (str): A string like "groupName:fieldName".
           isrc (int): The source number (1-based).

        Returns:
           Shot: A Shot object containing FD data.
        """

        try:
            import h5py
        except:
            print("h5py not found, skipping frequency-domain data")
            return None

        group_name, field = key.split(":")
        group = self.receiver_group(group_name)
        nrecv = group.size

        if isinstance(self.sampling, UniformSweepSampling):
            of = self.sampling.ofreq
            nf = self.sampling.nfreq
            f_max = self.sampling.f_max

            wavelet = self.source_group.signal(isrc)
            spectrum = wavelet.spectrum
        else:
            of = 0
            spectrum = np.ones([self.sampling.nfreq])

        u = np.zeros((nf, nrecv), dtype=np.csingle)

        # Loop over frequencies and load data
        for ifreq, freq in enumerate(self.sampling.freqs):
            file = os.path.join(group.directory, f"{group_name}_{ifreq}.h5")
            i_omega = np.csingle(1j * 2 * np.pi * freq)

            if ifreq >= of and not os.path.exists(file):
                warnings.warn(f"File {file} does not exist.", UserWarning)
            else:
                with h5py.File(file, "r") as f:
                    # Real + imaginary parts
                    u[ifreq, :] += np.csingle(1j) * f[f"{field}_{isrc}_im"][()]
                    u[ifreq, :] += f[f"{field}_{isrc}_re"][()]

                    # Apply wavelet
                    u[ifreq, :] *= spectrum[ifreq]

                    # For fiber-type receivers, multiply by iω for strain *rate*
                    if group.kind == "fiber":
                        u[ifreq, :] *= i_omega

                    f.close()

        return ShotRecord(
            type="FD",
            number=isrc,
            sampling=self.sampling,
            source=self.source(isrc),
            receiver_group=group,
            field=field,
            data=u,
        )

    def read_shot_TD(self, key: str, isrc: int) -> ShotRecord:
        """Read time-domain shot data by first reconstructing from the frequency-domain.

        Args:
           key (str): A string like "groupName:fieldName".
           isrc (int): The source number (1-based).

        Returns:
           Shot: A Shot object containing time-domain data.
        """

        if not isinstance(self.sampling, UniformSweepSampling):
            raise ValueError(
                "Time-domain data is only supported for uniform sweep sampling."
            )

        try:
            import pyfftw.interfaces.numpy_fft as fft
        except:
            print("pyfftw not found, using numpy for FFT (slow)")
            import numpy.fft as fft

        group_name, field = key.split(":")
        group = self.receiver_group(group_name)
        nrecv = group.size

        nf = self.sampling.nfreq
        nF = self.sampling.nFreq

        fd = self.read_shot_FD(key, isrc)

        # If upscaled, create a bigger array for inverse transform
        if nF > nf:
            FD = np.zeros((nF, nrecv), dtype=np.csingle)
            FD[:nf, :] = fd.data[:nf, :]
            del fd
            td = fft.irfft(FD, axis=0)
            del FD
        else:
            td = fft.irfft(fd.data, axis=0)
            del fd

        return ShotRecord(
            type="TD",
            number=isrc,
            sampling=self.sampling,
            source=self.source(isrc),
            receiver_group=group,
            field=field,
            data=td,
        )

    # def read_shot_TD(self, receiver_group: str, shot_number: int) -> ShotRecord:
    #    """Read a time-domain shot record.

    #    Args:
    #       receiver_group (str): Name of the receiver group.
    #       shot_number (int): Shot number.

    #    Returns:
    #       ShotRecord: The time-domain shot record.
    #    """
    #    if not isinstance(self.sampling, UniformSweepSampling):
    #       raise ValueError("Time-domain data requires UniformSweepSampling")

    #    group = self.acquisition.receiver_group(receiver_group)
    #    source = self.acquisition.source(shot_number)

    #    # Try to read frequency domain data first
    #    try:
    #       fd_record = self.read_shot_FD(receiver_group, shot_number)
    #    except:
    #       raise ValueError(f"Could not read FD data for receiver group '{receiver_group}' and shot {shot_number}")

    #    # Convert to time domain using FFT
    #    try:
    #       import pyfftw.interfaces.numpy_fft as fft
    #    except:
    #       warnings.warn('pyfftw not found, using numpy for FFT (slow)')
    #       import numpy.fft as fft

    #    nf = self.sampling.nfreq
    #    nF = self.sampling.nFreq

    #    # If upscaled, create a bigger array for inverse transform
    #    if nF > nf:
    #       FD = np.zeros((nF, group.size), dtype=np.csingle)
    #       FD[:nf, :] = fd_record.data[:nf, :]
    #       td = fft.irfft(FD, axis=0)
    #    else:
    #       td = fft.irfft(fd_record.data, axis=0)

    #    return ShotRecord(
    #       type="TD",
    #       number=shot_number,
    #       sampling=self.sampling,
    #       source=source,
    #       receiver_group=group,
    #       field=fd_record.field,
    #       data=td
    #    )

    # def read_shot_FD(self, receiver_group: str, shot_number: int) -> ShotRecord:
    #    """Read a frequency-domain shot record.

    #    Args:
    #       receiver_group (str): Name of the receiver group.
    #       shot_number (int): Shot number.

    #    Returns:
    #       ShotRecord: The frequency-domain shot record.
    #    """
    #    group = self.acquisition.receiver_group(receiver_group)
    #    source = self.acquisition.source(shot_number)
    #    nrecv = group.size
    #    nf = self.sampling.nfreq

    #    # Initialize complex data array
    #    u = np.zeros((nf, nrecv), dtype=np.csingle)

    #    # Loop over frequencies and load data
    #    for ifreq, freq in enumerate(self.sampling.freqs):
    #       file = Path(self.directory) / group.name / f"{group.name}_{ifreq}.h5"

    #       if not os.path.exists(file):
    #          warnings.warn(f"File {file} does not exist.", UserWarning)
    #          continue

    #       with h5py.File(file, "r") as f:
    #          # Real + imaginary parts
    #          field = group.field  # Assuming field is stored in receiver group
    #          u[ifreq, :] += np.csingle(1j) * f[f"{field}_{shot_number}_im"][()]
    #          u[ifreq, :] +=              f[f"{field}_{shot_number}_re"][()]

    #          # For fiber-type receivers, multiply by iω for strain *rate*
    #          if group.device._type == 'ReceiverFiber':
    #             i_omega = np.csingle(1j * 2 * np.pi * freq)
    #             u[ifreq, :] *= i_omega

    #    return ShotRecord(
    #       type="FD",
    #       number=shot_number,
    #       sampling=self.sampling,
    #       source=source,
    #       receiver_group=group,
    #       field=group.field,
    #       data=u
    #    )

    # def write_hdf5(self, filename: str, **kwargs):
    #    """Write shot records to an HDF5 file.

    #    Args:
    #       filename (str): Output HDF5 filename.
    #       **kwargs: Additional arguments passed to h5py.create_dataset().
    #    """
    #    with h5py.File(filename, 'w') as f:
    #       for receiver_group in self.acquisition.receiver_groups:
    #          grp = f.create_group(receiver_group.name)

    #          for ifreq, freq in enumerate(self.sampling.freqs):
    #             dset = grp.create_dataset(f"freq_{ifreq}",
    #                                     shape=(self.acquisition.num_sources, receiver_group.size),
    #                                     dtype=np.complex64,
    #                                     **kwargs)

    #             for isrc in range(self.acquisition.num_sources):
    #                try:
    #                   record = self.read_shot_FD(receiver_group.name, isrc+1)
    #                   dset[isrc,:] = record.data[ifreq,:]
    #                except:
    #                   warnings.warn(f"Could not read shot {isrc+1} for frequency {freq}")

    # def write_segy(self, filename: str, **kwargs):
    #    """Write shot records to a SEG-Y file.

    #    Args:
    #       filename (str): Output SEG-Y filename.
    #       **kwargs: Additional arguments passed to ShotRecord.write_segy().
    #    """
    #    for receiver_group in self.acquisition.receiver_groups:
    #       for isrc in range(self.acquisition.num_sources):
    #          try:
    #             record = self.read_shot_TD(receiver_group.name, isrc+1)
    #             segy_file = f"{filename}_{receiver_group.name}_shot{isrc+1}.sgy"
    #             record.write_segy(segy_file, **kwargs)
    #          except:
    #             warnings.warn(f"Could not write shot {isrc+1} for receiver group {receiver_group.name}")

    # def read_records(self, sorting: str = 'shot', **kwargs) -> ShotRecord:
    #    """Generator that yields records with the specified sorting.

    #    Args:
    #       sorting (str): Sorting order ('shot', 'cmp', 'crp', 'csp'). Default is 'shot'.
    #       **kwargs: Additional keyword arguments for sorting.

    #    Yields:
    #       ShotRecord: Next record in the specified sorting order.
    #    """
    #    if sorting == 'shot':
    #       yield from self._read_records_shot(**kwargs)
    #    elif sorting == 'cmp':
    #       yield from self._read_records_cmp(**kwargs)
    #    elif sorting == 'crp':
    #       yield from self._read_records_crp(**kwargs)
    #    elif sorting == 'csp':
    #       yield from self._read_records_csp(**kwargs)
    #    else:
    #       raise ValueError(f"Unsupported sorting order: {sorting}")

    # def _read_records_shot(self, shot_number: int):
    #    """Generator for records with common shot point sorting.

    #    Args:
    #       shot_number (int): Shot number.

    #    Yields:
    #       ShotRecord: Next record for the specified shot number.
    #    """
    #    for receiver_group in self.acquisition.receiver_groups:
    #       try:
    #          yield self.read_shot_FD(receiver_group.name, shot_number)
    #       except:
    #          warnings.warn(f"Could not read shot {shot_number} for receiver group {receiver_group.name}")

    # def _read_records_cmp(self, cmp_x: float):
    #    """Generator for records with common midpoint sorting.

    #    Args:
    #       cmp_x (float): X-coordinate of common midpoint.

    #    Yields:
    #       ShotRecord: Next record containing the specified midpoint.
    #    """
    #    for isrc in range(self.acquisition.num_sources):
    #       source = self.acquisition.source(isrc + 1)
    #       for receiver_group in self.acquisition.receiver_groups:
    #          group = self.acquisition.receiver_group(receiver_group)
    #          for i, receiver in enumerate(group.receivers):
    #             midpoint = (source.coord[0] + receiver.coord[0]) / 2
    #             if abs(midpoint - cmp_x) < 1e-6:
    #                try:
    #                   yield self.read_shot_FD(receiver_group, isrc + 1)
    #                except:
    #                   warnings.warn(f"Could not read shot {isrc+1} for receiver group {receiver_group}")
    #                break

    # def _read_records_crp(self, receiver_x: float):
    #    """Generator for records with common receiver point sorting.

    #    Args:
    #       receiver_x (float): X-coordinate of receiver point.

    #    Yields:
    #       ShotRecord: Next record containing the specified receiver point.
    #    """
    #    for receiver_group in self.acquisition.receiver_groups:
    #       group = self.acquisition.receiver_group(receiver_group)
    #       for i, receiver in enumerate(group.receivers):
    #          if abs(receiver.coord[0] - receiver_x) < 1e-6:
    #             for isrc in range(self.acquisition.num_sources):
    #                try:
    #                   yield self.read_shot_FD(receiver_group, isrc + 1)
    #                except:
    #                   warnings.warn(f"Could not read shot {isrc+1} for receiver group {receiver_group}")

    # def _read_records_csp(self, source_x: float):
    #    """Generator for records with common source point sorting.

    #    Args:
    #       source_x (float): X-coordinate of source point.

    #    Yields:
    #       ShotRecord: Next record containing the specified source point.
    #    """
    #    for isrc in range(self.acquisition.num_sources):
    #       source = self.acquisition.source(isrc + 1)
    #       if abs(source.coord[0] - source_x) < 1e-6:
    #          for receiver_group in self.acquisition.receiver_groups:
    #             try:
    #                yield self.read_shot_FD(receiver_group, isrc + 1)
    #             except:
    #                warnings.warn(f"Could not read shot {isrc+1} for receiver group {receiver_group}")

    # def interpolate_records(self, receiver_group: str, shot_number: int) -> ShotRecord:
    #    """Interpolate records similar to SignalFromFile.

    #    Args:
    #       receiver_group (str): Name of the receiver group.
    #       shot_number (int): Shot number.

    #    Returns:
    #       ShotRecord: Interpolated shot record.
    #    """
    #    # Read the original shot record
    #    original_record = self.read_shot_TD(receiver_group, shot_number)

    #    # Perform linear interpolation on the data
    #    interpolated_data = np.interp(
    #       np.arange(0, original_record.data.shape[0], 0.5),  # New time points
    #       np.arange(original_record.data.shape[0]),          # Original time points
    #       original_record.data                               # Original data
    #    )

    #    return ShotRecord(
    #       type="TD",
    #       number=shot_number,
    #       sampling=self.sampling,
    #       source=original_record.source,
    #       receiver_group=original_record.receiver_group,
    #       field=original_record.field,
    #       data=interpolated_data
    #    )

    # def build_database(self, file_map: Dict[str, str]):
    #    """Build a database from various files.

    #    Args:
    #       file_map (Dict[str, str]): Mapping of receiver group names to file paths.
    #    """
    #    for receiver_group, file_pattern in file_map.items():
    #       # Extract shot number and component from the file pattern
    #       shot_pattern = re.search(r'{shot}', file_pattern)
    #       comp_pattern = re.search(r'{comp}', file_pattern)

    #       if not shot_pattern or not comp_pattern:
    #          warnings.warn(f"Invalid file pattern for receiver group {receiver_group}: {file_pattern}")
    #          continue

    #       # Iterate over possible shot numbers and components
    #       for shot_number in range(1, self.acquisition.num_sources + 1):
    #          for comp in ['x', 'y', 'z']:  # Assuming components are x, y, z
    #             file_path = file_pattern.format(shot=shot_number, comp=comp)
    #             if not os.path.exists(file_path):
    #                warnings.warn(f"File {file_path} does not exist for receiver group {receiver_group}")
    #                continue

    #             # Load data from the file and add to the database
    #             with h5py.File(file_path, 'r') as f:
    #                data = f[f"{receiver_group}_{shot_number}_{comp}"][:]
    #                # Store data in the database (implementation depends on your database structure)
    #                # Example: self.store_data(receiver_group, shot_number, comp, data)
