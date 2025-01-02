from langchain_huggingface import HuggingFaceEmbeddings
from transformers import PreTrainedModel, PreTrainedTokenizerFast, ReactCodeAgent

from webwise.agents import WikiAgent, ContentExtractor, CustomWebAgent
from webwise.config import (
    LLMConfig, 
    EmbeddingsConfig, 
    ContentExtractionConfig, 
    WebSearchConfig, 
    WebSourceType, 
    WebSourceConfig
)
from webwise.llm import LLMEngine
from webwise import prompts


class WebWiseSystem:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        search_cfg: WebSearchConfig,
        llm_cfg: LLMConfig = LLMConfig(),
        embeddings_cfg: EmbeddingsConfig = EmbeddingsConfig(),
        content_cfg: ContentExtractionConfig = ContentExtractionConfig()
    ):
        self.llm = LLMEngine(model, tokenizer, llm_cfg)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_cfg.model_name,
            model_kwargs=embeddings_cfg.model_kwargs,
            encode_kwargs=embeddings_cfg.encoding_kwargs
        )
        
        # Initialize agents
        self.agents = {}
        for source in search_cfg.sources:
            if source.source_type == WebSourceType.WIKI:
                self.agents[source.source_type] = WikiAgent(
                    llm=self.llm,
                    embeddings=self.embeddings,
                    content_cfg=content_cfg,
                    system_prompt=source.system_prompt or prompts.WIKI_PROMPT
                )
            elif source.source_type == WebSourceType.CUSTOM:
                self.agents[source.source_type] = CustomWebAgent(
                    llm=self.llm,
                    embeddings=self.embeddings,
                    content_cfg=content_cfg,
                    base_url=source.base_url,
                    system_prompt=source.system_prompt or prompts.CUSTOM_SITE_PROMPT
                )
        
        self.content_extractor = ContentExtractor(
            llm=self.llm,
            embeddings=self.embeddings,
            content_cfg=content_cfg
        )
        
        self.search_config = search_cfg

    def process_query(self, query: str) -> str:
        """Process a query using the configured agents and return a response."""
        results = []
        
        for agent in self.agents.values():
            agent_results = agent.search(query, max_results=self.search_config.max_results)
            results.extend(agent_results)
            
        if not results:
            return "No relevant information found."
            
        # Extract and synthesize content
        extracted_content = self.content_extractor.process(results)
        
        # Generate final response
        response = self.llm.generate(
            prompt=prompts.RESPONSE_TEMPLATE.format(
                content=extracted_content,
                query=query,
                sources="\n".join(r.source for r in results)
            )
        )
        
        return response 