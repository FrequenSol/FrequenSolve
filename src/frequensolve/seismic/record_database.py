"""
Not used yet; this will complement survey.py for reading and visualizing
data when finished.

Right now this is just a hodge-podge of code that was displaced in
the refactoring process.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import h5py
import numpy as np

from frequensolve.seismic.shot_record import Record
from frequensolve.simulation.sampling import UniformSweepSampling


@dataclass
class RecordDatabase:
    metadata: Dict[str, Any]
    records: List[str]
    _upscale: int

    def __init__(self, metadata: Dict[str, Any], records: List[str], upscale: int = 1):
        self.metadata = metadata
        self.records = records
        self.upscale = upscale

    @classmethod
    def from_results(cls, results: dict, proj_path: Union[str, Path], upscale: int = 1):
        """Create a RecordDatabase from a dictionary of results.

        Args:
            results: A dictionary of results from a Frontera job.
            proj_path: The path to the project directory.

        Returns:
            A RecordDatabase object.
        """
        proj_path = Path(proj_path).resolve()

        f_map = results["frequencies"]
        for key, value in f_map.items():
            f_map[key] = float(value)
        meta = {
            "project": proj_path,
            "simulation": proj_path / results["simulation"],
            "df": float(f_map[2] - f_map[1]) if len(f_map) > 1 else 0.0,
            "f_max": (
                float(np.max(list(f_map.values()))) if len(f_map) > 1 else f_map[1]
            ),
            "f_map": f_map if len(f_map) > 1 else {},
        }

        records = []
        for file, comps in results["datasets"].items():
            file = Path(file)
            parts = file.name.split("_")

            fbase = str(proj_path / file.parent / "_".join(parts[:-1])) + "_[ifreq].h5"

            for comp in comps:
                record = fbase + f":{comp}"
                if record not in records:
                    records.append(record)

        return cls(metadata=meta, records=records, upscale=upscale)

    @property
    def upscale(self) -> int:
        return self._upscale

    @upscale.setter
    def upscale(self, upscale: int) -> None:
        self._upscale = upscale

    def times(self, upscale: Optional[int] = None) -> np.ndarray:
        """Returns the times of the records."""

        upscale = self.upscale if upscale is None else upscale
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        return sampling.T_list

    def __len__(self) -> int:
        """Returns the number of records in the database."""
        return len(self.records)

    def __iter__(self):
        for record in self.records:
            yield Record(record, self.metadata, self._upscale)

    def __getitem__(self, key: int):
        return Record(self.records[key], self.metadata, self._upscale)

    def __str__(self) -> str:
        meta_str = "Metadata:\n"
        meta_str += f"  Simulation: {self.metadata['simulation']}\n"
        meta_str += f"  Frequency step (df): {self.metadata['df']:.2f} Hz\n"
        meta_str += f"  Maximum frequency: {self.metadata['f_max']:.2f} Hz\n"
        meta_str += "  Frequency map: "
        for k, v in self.metadata["f_map"].items():
            meta_str += f"{k}:{v:.2f}, "
        meta_str = meta_str[:-2] + "\n"

        records_str = "\nRecords:\n"
        for i, record in enumerate(self.records):
            records_str += f"  {i}: {record}\n"

        return meta_str + records_str

    def consolidate_h5(self):
        """Consolidate records into single h5 files to improve efficiency and convenience."""
        if self.is_consolidated:
            return

        n_freq = len(self.metadata["f_map"])

        # Get unique receiver group bases
        bases = set()
        for file in self.records:
            fbase = "_".join(file.split(":")[0].split("_")[:-1])
            bases.add(fbase)

        for base in bases:
            base_files = [f for f in self.records if f.startswith(base)]

            fields = set()
            for file in base_files:
                _, comp = file.split(":")
                fields.add("_".join(comp.split("_")[:-1]))

            print(f"Consolidating {base}")

            # Create new consolidated HDF5 file
            new_file = f"{base}_consolidated.h5"

            if os.path.exists(new_file):
                access = "r+"
            else:
                access = "w"

            with h5py.File(new_file, access) as f:
                sample_file = base_files[0]
                _, comp = sample_file.split(":")

                # For each field component
                for field in fields:
                    if field in f:
                        field_group = f[field]
                    else:
                        field_group = f.create_group(field)

                    # Create source subgroups
                    source_files = [f for f in base_files if f":{field}" in f]
                    for src_file in source_files:
                        # Get source number
                        _, comp = src_file.split(":")
                        isrc = int(comp.split("_")[-1])

                        # Create source group
                        if f"source_{isrc}" in field_group:
                            src = field_group[f"source_{isrc}"]
                        else:
                            src = field_group.create_group(f"source_{isrc}")

                        fbase = "_".join(src_file.split(":")[0].split("_")[:-1])

                        with h5py.File(f"{fbase}_1.h5", "r") as freq_file:
                            shape = freq_file[f"{field}_{isrc}_re"].shape
                            n_recv = shape[0]

                        # If dataset exists, make sure it has the correct shape
                        if "real" in src:
                            nf, nr = src["real"].shape

                            if nf < n_freq or nr != n_recv:
                                del src["real"]
                                del src["imag"]

                        # If dataset does not exist, create it
                        if "real" not in src:
                            src.create_dataset(
                                "real", (n_freq,) + (n_recv,), dtype=np.float32
                            )
                            src.create_dataset(
                                "imag", (n_freq,) + (n_recv,), dtype=np.float32
                            )

                        # Read and store data from individual frequency files
                        for i in range(n_freq):
                            freq_file = f"{fbase}_{i+1}.h5"
                            if os.path.exists(freq_file):
                                with h5py.File(freq_file, "r") as ff:
                                    src["real"][i, :] = ff[f"{field}_{isrc}_re"][:]
                                    src["imag"][i, :] = ff[f"{field}_{isrc}_im"][:]

            # Delete individual frequency files after consolidation
            for i in range(n_freq):
                freq_file = f"{fbase}_{i+1}.h5"
                if os.path.exists(freq_file):
                    os.remove(freq_file)

    @property
    def is_consolidated(self) -> bool:
        """Check if records have been consolidated into single h5 files.

        Returns:
            bool: True if all records have been consolidated, False otherwise.
        """

        n_freq = len(self.metadata["f_map"])

        # Get unique receiver group bases
        bases = set()
        for file in self.records:
            fbase = "_".join(file.split(":")[0].split("_")[:-1])
            bases.add(fbase)

        for base in bases:
            base_files = [f for f in self.records if f.startswith(base)]
            for ifreq in range(1, n_freq + 1):
                file = f"{base}_{ifreq}.h5"
                if os.path.exists(file):
                    return False
        return True

    def read_consolidated(self, group: str, field: str, source: int) -> np.ndarray:
        """Read data from a consolidated HDF5 file.

        Args:
            group (str): Name of the receiver group
            field (str): Field component to read (e.g. 'u_x', 'u_z')
            source (int): Source number

        Returns:
            np.ndarray: Complex array containing the frequency domain data
                       with shape (n_frequencies, n_receivers)

        Raises:
            FileNotFoundError: If consolidated file does not exist
            KeyError: If requested data not found in file
        """
        consolidated_file = f"{group}_consolidated.h5"
        if not os.path.exists(consolidated_file):
            raise FileNotFoundError(f"Consolidated file {consolidated_file} not found")

        with h5py.File(consolidated_file, "r") as f:
            if field not in f:
                raise KeyError(f"Field {field} not found in {consolidated_file}")

            source_group = f[field][f"source_{source}"]
            real_data = source_group["real"][:]
            imag_data = source_group["imag"][:]

            return real_data + 1j * imag_data

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
