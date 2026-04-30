"""Preferred trace-record imports.

The implementation currently lives in ``shot_record`` to preserve existing
imports. New code should import ``TraceRecord`` from this module.
"""

from frequensolve.seismic.shot_record import ShotRecord, TraceRecord, array_to_segy

__all__ = ["TraceRecord", "ShotRecord", "array_to_segy"]
