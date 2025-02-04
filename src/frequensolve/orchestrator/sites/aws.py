from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..jobs.base_job import BaseJob
from .base_site import BaseSite, BaseSiteConfig

__all__ = ["AWSSiteConfig", "AWSSite"]


@dataclass
class AWSSiteConfig(BaseSiteConfig):
    """Cluster configuration for job execution on AWS EC2.

    This class defines resource requirements and constraints for job execution
    on AWS EC2 instances.

    Args:
       instance_type: EC2 instance type (e.g. 't2.micro', 'c5.xlarge').
       max_duration: Maximum time resources can be requested (HH:MM:SS).
       memory_per_rank: Memory allocation per MPI rank in megabytes.

    Attributes:
       instance_type:    EC2 instance type.
       max_duration:     Maximum time resources can be requested (HH:MM:SS).
       memory_per_rank:  Memory allocation per MPI rank in megabytes.
    """

    instance_type: str
    region: Optional[str] = None
    max_duration: Optional[str] = None
    config_file: Optional[Union[str, Path]] = None

    # TODO: figure out what info I need about site, etc.
    def __post_init__(self):
        import boto3

        if self.config_file is not None:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            self.region = config["region"]

        my_config = boto3.Config(
            region_name=self.region,
            signature_version="v4",
        )

        # Get information about instance type
        client = boto3.client("ec2", config=my_config)
        response = client.describe_instance_types(InstanceTypes=[self.instance_type])

        # Example info I can get
        if response["InstanceTypes"]:
            info = response["InstanceTypes"][0]
            vcpus = info["VCpuInfo"]["DefaultVCpus"]
            memory_gb = info["MemoryInfo"]["SizeInMiB"] / 1024
            if "GpuInfo" in info:
                gpus = sum(gpu["Count"] for gpu in info["GpuInfo"]["Gpus"])
            else:
                gpus = 0

    @classmethod
    def from_dict(cls, data: dict) -> "AWSSiteConfig":
        return cls(**data)

    def __dict__(self) -> dict:
        return {
            "instance_type": self.instance_type,
            "region": self.region,
            "max_duration": self.max_duration,
        }

    # TODO: Instead of JSON, use INI-formatted file,
    #       see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
    @classmethod
    def load(cls, name: str) -> "AWSSiteConfig":
        name += ".json" if not name.endswith(".json") else ""
        try:
            with open(name, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {name}: {e}")
        return cls.from_dict(data)

    def save(self, name: str) -> None:
        name += ".json" if not name.endswith(".json") else ""
        try:
            with open(name, "w") as f:
                json.dump(self.__dict__(), f, indent=3)
        except Exception as e:
            warnings.warn(f"Failed to save configuration to {name}: {e}")


class AWSSite(BaseSite):
    """AWS site configuration."""

    def __init__(self, config: AWSSiteConfig):
        self.config = config

    # In this case we can create a static
    def provision(self):
        pass

    def deprovision(self):
        pass

    def check_status(self):
        pass

    def wait_provisioned(self):
        pass

    def submit(job: BaseJob):
        pass
