from __future__ import annotations

from ..types import StructuredDatasetPreset
from .enterprise_ticket import PRESET as ENTERPRISE_TICKET_PRESET
from .restaurant_review import PRESET as RESTAURANT_REVIEW_PRESET
from .social_media import PRESET as SOCIAL_MEDIA_PRESET

PRESETS: dict[str, StructuredDatasetPreset] = {
    ENTERPRISE_TICKET_PRESET.name: ENTERPRISE_TICKET_PRESET,
    RESTAURANT_REVIEW_PRESET.name: RESTAURANT_REVIEW_PRESET,
    SOCIAL_MEDIA_PRESET.name: SOCIAL_MEDIA_PRESET,
}
