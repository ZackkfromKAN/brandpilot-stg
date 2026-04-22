from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .client import BrandPilotClient


class BrandContext(BaseModel):
    """Live brand data loaded from BrandPilot Backend at run start."""
    brand_id:     str
    account_id:   str
    brand_name:   str         = ""
    brand_manual: Dict[str, Any] = Field(default_factory=dict)
    passport:     Dict[str, Any] = Field(default_factory=dict)
    markets:      List[Dict[str, Any]] = Field(default_factory=list)


class BaseRunConfig(BaseModel):
    """
    Auth + scope injected by the caller at run start.
    All fields are optional so legacy callers without API integration keep working.
    """
    cognito_token: Optional[str] = Field(default=None, exclude=True)
    account_id:    Optional[str] = Field(default=None, exclude=True)
    brand_id:      Optional[str] = Field(default=None, exclude=True)
    environment:   str           = Field(default="prod", exclude=True)

    # Populated by load_brand_context_node when API credentials are present
    brand_context: Optional[BrandContext] = Field(default=None, exclude=True)

    def api_client(self) -> Optional[BrandPilotClient]:
        """Returns a scoped API client if credentials are present, else None."""
        if self.cognito_token and self.account_id and self.brand_id:
            return BrandPilotClient(
                cognito_token=self.cognito_token,
                account_id=self.account_id,
                brand_id=self.brand_id,
                env=self.environment,
            )
        return None

    def has_api_credentials(self) -> bool:
        return bool(self.cognito_token and self.account_id and self.brand_id)
