from enum import Enum
from typing import Optional
from pydantic.dataclasses import dataclass
from . import prompts


class WebSourceType(Enum):
    WIKI = "wikipedia"
    CUSTOM = "custom"


@dataclass
class WebSourceConfig:
    source_type: WebSourceType
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None

    def __post_init__(self):
        if self.system_prompt is None:
            if self.source_type == WebSourceType.WIKI:
                self.system_prompt = prompts.WIKI_PROMPT
            else:
                self.system_prompt = prompts.DEFAULT_SYSTEM_PROMPT


@dataclass
class WebSearchConfig:
    sources: list[WebSourceConfig]
    max_results: int = 5
    timeout: int = 10 
