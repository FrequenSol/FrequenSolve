"""Run-record helpers mixed into FrequenSolve jobs.

These methods keep local metadata about where a job was staged or submitted,
including scheduler ids and site reconstruction hints, so later sessions can
inspect status and fetch outputs without rebuilding the job context manually.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

from frequensolve.simulation.jobs.artifacts import RunMetadata

if TYPE_CHECKING:
    from frequensolve.simulation.jobs.base import JobRecord


class JobRecordMixin:
    """Accessors for local run metadata and recorded site submissions.

    The mixin is used by ``BaseJob`` to read run status, remember remote site
    submissions, and fetch outputs through a recorded or explicit site object.
    """

    @property
    def run_state_file(self) -> Path:
        """Return the path to the Python-side run state summary.

        Returns:
            Path to ``_fs_python_run.json`` in the job result directory.
        """

        return self._result_path / "_fs_python_run.json"

    @property
    def run_records_file(self) -> Path:
        """Return the path to recorded local or remote run locations.

        Returns:
            Path to the run-record JSON file under ``results/_fs_run``.
        """

        return self._result_path / "_fs_run" / "runs.json"

    @property
    def run_metadata(self) -> RunMetadata:
        """Return structured solver and Python metadata beside results.

        Returns:
            ``RunMetadata`` read from this job's result directory.
        """

        return RunMetadata.read(self._result_path)

    def run_records(self) -> List[JobRecord]:
        """Return saved locations where this job has been staged or run.

        Returns:
            List of valid ``JobRecord`` entries. Corrupt or incomplete records
            are ignored so metadata inspection stays robust.
        """

        from frequensolve.simulation.jobs.base import JobRecord

        path = self.run_records_file
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        raw_records = payload.get("runs") if isinstance(payload, Mapping) else None
        if raw_records is None and isinstance(payload, list):
            raw_records = payload
        if not isinstance(raw_records, list):
            return []
        records = []
        for record in raw_records:
            if not isinstance(record, Mapping):
                continue
            try:
                records.append(JobRecord.from_fs(record))
            except (KeyError, TypeError, ValueError):
                continue
        return records

    def latest_run(self, site: Optional[str] = None) -> Optional[JobRecord]:
        """Return the most recently recorded run location.

        Args:
            site: Optional site name used to filter records.

        Returns:
            Latest matching ``JobRecord``, or ``None`` when no record exists.
        """

        records = self.run_records()
        if site is not None:
            records = [record for record in records if record.site == site]
        if not records:
            return None
        return sorted(
            records,
            key=lambda record: record.updated_at or record.submitted_at or "",
        )[-1]

    def write_run_record(self, record: JobRecord) -> JobRecord:
        """Persist or update a run location record.

        Args:
            record: Record to insert or replace. Existing records are matched
                by site, job directory, and scheduler id.

        Returns:
            The record that was written.
        """

        def key(item: JobRecord) -> tuple:
            return (item.site, str(item.job_dir), item.scheduler_id or "")

        records = [item for item in self.run_records() if key(item) != key(record)]
        records.append(record)
        payload = {
            "schema": "fs-job-run-records-1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": [item.to_fs() for item in records],
        }
        self._write_json_file(self.run_records_file, payload)
        return record

    def record_site_run(
        self,
        *,
        site: str,
        work_dir: Union[str, Path],
        scheduler_id: Optional[str] = None,
        status: str = "submitted",
        site_module: Optional[str] = None,
        site_class: Optional[str] = None,
        rel_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JobRecord:
        """Record the layout used by a site submission.

        Args:
            site: Site name.
            work_dir: Site work directory used as the project root.
            scheduler_id: Optional scheduler job id.
            status: Initial status to store on the record.
            site_module: Optional module path used to recreate the site.
            site_class: Optional class name used to recreate the site.
            rel_path: Optional site configuration path.
            metadata: Additional site-specific metadata.

        Returns:
            The persisted ``JobRecord``.
        """

        from frequensolve.simulation.jobs.base import JobLayout, JobRecord

        if self._file is None or self.simulation._file is None:
            self.save()
        now = datetime.now(timezone.utc).isoformat()
        layout = JobLayout.from_job(self, work_dir)
        record = JobRecord(
            site=site,
            work_dir=Path(work_dir),
            project_path=layout.project,
            job_dir=layout.job_dir,
            job_file=layout.job_file,
            result_dir=layout.result_dir,
            logs_dir=layout.logs_dir,
            scheduler_id=str(scheduler_id) if scheduler_id is not None else None,
            status=status,
            submitted_at=now,
            updated_at=now,
            fingerprint=self.fingerprint(),
            fingerprint_payload=self.fingerprint_payload(),
            site_module=site_module,
            site_class=site_class,
            rel_path=str(rel_path) if rel_path is not None else None,
            metadata=dict(metadata or {}),
        )
        return self.write_run_record(record)

    def fetch_traces(self, site=None, upscale: int = 1):
        """Fetch receiver traces from a recorded run location.

        Args:
            site: Optional initialized site object. When omitted, the latest
                recorded run must contain enough metadata to recreate its site.
            upscale: Trace upscaling factor passed to the site fetcher.

        Returns:
            Site-specific fetch result.
        """

        return self._resolve_fetch_site(site).fetch_traces(self, upscale=upscale)

    def fetch_wavefields(self, site=None, upscale: int = 1):
        """Fetch wavefield outputs from a recorded run location.

        Args:
            site: Optional initialized site object. When omitted, the latest
                recorded run is used.
            upscale: Wavefield upscaling factor passed to the site fetcher.

        Returns:
            Site-specific fetch result.
        """

        return self._resolve_fetch_site(site).fetch_wavefields(self, upscale=upscale)

    def fetch_outputs(self, site=None):
        """Fetch common output artifacts from a recorded run location.

        Args:
            site: Optional initialized site object. When omitted, the latest
                recorded run is used.

        Returns:
            Site-specific fetch result.
        """

        return self._resolve_fetch_site(site).fetch_outputs(self)

    def fetch_run_metadata(self, site=None):
        """Fetch run metadata from a recorded run location.

        Args:
            site: Optional initialized site object. When omitted, the latest
                recorded run is used.

        Returns:
            Site-specific fetch result.
        """

        return self._resolve_fetch_site(site).fetch_run_metadata(self)

    def fetch_logs(self, site=None, **kwargs):
        """Fetch logs from a recorded run location.

        Args:
            site: Optional initialized site object. When omitted, the latest
                recorded run is used.
            **kwargs: Additional options forwarded to the site log fetcher.

        Returns:
            Site-specific fetch result.
        """

        return self._resolve_fetch_site(site).fetch_logs(self, **kwargs)

    def _site_from_run_record(self, record: JobRecord):
        config_path = record.metadata.get("site_config_path")
        profile = record.metadata.get("site_profile")
        if config_path and profile:
            from frequensolve.orchestrator.sites.config_file import Site

            return Site(config_path=config_path, profile=profile)
        if not record.site_module or not record.site_class or not record.rel_path:
            raise ValueError(
                f"Run record for {record.site} cannot recreate a site; "
                "fetch through an initialized site instead."
            )
        module = importlib.import_module(record.site_module)
        site_class = getattr(module, record.site_class)
        return site_class(record.rel_path)

    def _resolve_fetch_site(self, site=None):
        if site is not None:
            return site
        record = self.latest_run()
        if record is None:
            raise ValueError(
                "This job has no recorded remote run. Submit it once or pass a site."
            )
        return self._site_from_run_record(record)
