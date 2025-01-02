from dataclasses import field
from enum import Enum
from typing import Optional, List
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
class ContentExtractionConfig:
    chunk_size: int = 512
    chunk_overlap: int = 256
    passage_count: int = 5
    text_separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", ",", " ", ""])
    max_retries: int = 3


@dataclass
class WebSearchConfig:
    sources: List[WebSourceConfig] = field(default_factory=lambda: [
        WebSourceConfig(source_type=WebSourceType.WIKI)
    ])
    max_results: int = 5
    timeout: int = 10


@dataclass
class EmbeddingsConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs: dict = field(default_factory=dict)
    encoding_kwargs: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    use_history_summary: bool = True
    max_tokens: int = 1024
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False 