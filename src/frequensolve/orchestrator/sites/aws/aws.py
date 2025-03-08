import json
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import boto3

from frequensolve.orchestrator.config.base import BaseSiteConfig
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.orchestrator.tasks.base import BaseTask

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

        # my_config = boto3.Config(
        #     region_name=self.region,
        #     signature_version="v4",
        # )

        # # Get information about instance type
        # client = boto3.client("ec2", config=my_config)
        # response = client.describe_instance_types(InstanceTypes=[self.instance_type])

        # # Example info I can get
        # if response["InstanceTypes"]:
        #     info = response["InstanceTypes"][0]
        #     vcpus = info["VCpuInfo"]["DefaultVCpus"]
        #     memory_gb = info["MemoryInfo"]["SizeInMiB"] / 1024
        #     if "GpuInfo" in info:
        #         gpus = sum(gpu["Count"] for gpu in info["GpuInfo"]["Gpus"])
        #     else:
        #         gpus = 0

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
        self.cf_client = boto3.client("cloudformation", region_name=self.config.region)

    # In this case we can create a static
    def provision(self):
        # Get template path
        template_path = (
            Path(__file__).parent / "assets" / "cloudformation" / "cluster.yaml"
        )

        # Read template
        with open(template_path) as f:
            template_body = f.read()

        # Create stack name
        stack_name = f"pcs-cluster-{uuid.uuid4().hex[:8]}"

        try:
            # Create stack
            response = self.cf_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=[
                    {
                        "ParameterKey": "KeyName",
                        "ParameterValue": "quick-start-pcs",  # TODO: Automate key creation
                    },
                    {
                        "ParameterKey": "NodeArchitecture",
                        "ParameterValue": "Graviton",
                    },
                    {
                        "ParameterKey": "SlurmVersion",
                        "ParameterValue": "24.05",
                    },
                ],
                Capabilities=[
                    "CAPABILITY_IAM",
                    "CAPABILITY_NAMED_IAM",
                    "CAPABILITY_AUTO_EXPAND",
                ],
                OnFailure="DELETE",
            )

            # Store stack ID
            self.stack_id = response["StackId"]

        except Exception as e:
            raise RuntimeError(f"Failed to create CloudFormation stack: {str(e)}")

    def deprovision(self):
        try:
            # Delete the CloudFormation stack
            self.cf_client.delete_stack(StackName=self.stack_id)

            # Wait for stack deletion to complete
            waiter = self.cf_client.get_waiter("stack_delete_complete")
            waiter.wait(
                StackName=self.stack_id, WaiterConfig={"Delay": 30, "MaxAttempts": 60}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to delete CloudFormation stack: {str(e)}")

    def check_status(self):
        pass

    def wait_provisioned(self):
        try:
            # Wait for stack creation to complete
            waiter = self.cf_client.get_waiter("stack_create_complete")
            waiter.wait(
                StackName=self.stack_id, WaiterConfig={"Delay": 30, "MaxAttempts": 60}
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed while waiting for CloudFormation stack creation: {str(e)}"
            )

    def submit(job: BaseTask):
        pass

    def cancel_job(self, job_id: str) -> None:
        pass


if __name__ == "__main__":
    site = AWSSite(AWSSiteConfig(instance_type="t2.micro", region="us-east-1"))
    site.provision()
    provision_started = time.time()
    print(f"Provisioning stack: {site.stack_id}")
    site.wait_provisioned()
    provision_duration = time.time() - provision_started
    print(f"Stack {site.stack_id} provisioned in {provision_duration:.2f} seconds")
    print("waiting 60 seconds")
    time.sleep(60)
    deprovision_started = time.time()
    print(f"Deprovisioning stack: {site.stack_id}")
    site.deprovision()
    deprovision_duration = time.time() - deprovision_started
    print(f"Stack {site.stack_id} deprovisioned in {deprovision_duration:.2f} seconds")
