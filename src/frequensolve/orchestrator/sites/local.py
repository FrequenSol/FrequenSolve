from dataclasses import dataclass

from .base_site import *

__all__ = ["LocalSiteConfig", "LocalSite"]


@dataclass
class LocalSiteConfig(BaseSiteConfig):
    """Local site configuration."""


class LocalSite(BaseSite):
    """Local site configuration."""

    def __init__(self, config: LocalSiteConfig):
        self.config = config
        self.status = SiteStatus(status="unknown")
