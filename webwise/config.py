from dataclasses import field
from importlib import resources
from enum import Enum
from typing import Optional
from webwise import prompts
from pydantic.dataclasses import dataclass


class WebSourceType(Enum):
    WIKI = "wiki"
    CUSTOM = "custom"
    NEWS = "news"


@dataclass
class WebSourceConfig:
    source_type: WebSourceType = WebSourceType.WIKI
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class ContentExtractionConfig:
    chunk_size: int = 512
    chunk_overlap: int = 256
    passage_count: int = 5
    text_separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", ",", " ", ""])
    max_retries: int = 3


@dataclass
class WebSearchConfig:
    results_per_source: int = 5
    max_content_length: int = 1000
    sources: list[WebSourceConfig] = field(default_factory=lambda: [WebSourceConfig()])


@dataclass
class EmbeddingsConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs: dict = field(default_factory=dict)
    encoding_kwargs: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    use_history_summary: bool = True
    max_tokens: int = 1024
    temperature: float|None = None
    top_k: int|None = None
    top_p: float|None = None
    do_sample: bool = False


@dataclass
class SystemPrompts:
    coordinator_prompt: str = resources.read_text(prompts, 'coordinator_prompt.txt')
    wiki_search_prompt: str = resources.read_text(prompts, 'wiki_search_prompt.txt')
    content_extractor_prompt: str = resources.read_text(prompts, 'content_extractor_prompt.txt') 