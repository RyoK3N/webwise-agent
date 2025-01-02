from langchain_huggingface import HuggingFaceEmbeddings
from transformers import PreTrainedModel, PreTrainedTokenizerFast, ReactCodeAgent

from webwise.agents import WikiAgent, ContentExtractor, CustomWebAgent
from webwise.config import (LLMConfig, EmbeddingsConfig, SystemPrompts, ContentExtractionConfig, 
                         WebSearchConfig, WebSourceType, WebSourceConfig)
from webwise.llm import LLMEngine
from webwise.utils import format_agent_logs


class WebWiseSystem:
    """Main system class that coordinates multiple agents for web-based information retrieval."""
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerFast,
                 prompts: SystemPrompts = SystemPrompts(),
                 llm_cfg: LLMConfig = LLMConfig(),
                 embeddings_cfg: EmbeddingsConfig = EmbeddingsConfig(),
                 extraction_cfg: ContentExtractionConfig = ContentExtractionConfig(),
                 search_cfg: WebSearchConfig = WebSearchConfig()):
        
        # Initialize the language model engine
        self.llm_engine = LLMEngine(
            model=model,
            tokenizer=tokenizer,
            use_history_summary=llm_cfg.use_history_summary,
            max_new_tokens=llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
            top_k=llm_cfg.top_k,
            top_p=llm_cfg.top_p,
            do_sample=llm_cfg.do_sample
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_cfg.model_name,
            **embeddings_cfg.model_kwargs
        )
        
        # Initialize content extractor
        self.content_extractor = ContentExtractor(
            llm_engine=self.llm_engine,
            embeddings=self.embeddings,
            system_prompt=prompts.content_extractor_prompt,
            extraction_cfg=extraction_cfg
        )
        
        # Initialize web source agents
        self.agents = []
        for source_config in search_cfg.sources:
            if source_config.source_type == WebSourceType.WIKI:
                agent = WikiAgent(
                    llm_engine=self.llm_engine,
                    content_extractor=self.content_extractor,
                    system_prompt=prompts.wiki_search_prompt,
                    search_cfg=search_cfg
                )
            elif source_config.source_type == WebSourceType.CUSTOM:
                if not source_config.base_url:
                    raise ValueError("base_url is required for custom website source")
                system_prompt = (source_config.system_prompt or 
                               prompts.wiki_search_prompt)  # fallback to wiki prompt
                agent = CustomWebAgent(
                    llm_engine=self.llm_engine,
                    content_extractor=self.content_extractor,
                    system_prompt=system_prompt,
                    base_url=source_config.base_url,
                    search_cfg=search_cfg
                )
            self.agents.append(agent)

        # Initialize coordinator agent with all source tools
        agent_tools = [agent.as_tool() for agent in self.agents]
        self.coordinator = ReactCodeAgent(
            system_prompt=prompts.coordinator_prompt,
            llm_engine=self.llm_engine,
            tools=agent_tools
        )

    @property
    def coordinator_logs(self) -> dict[str, any]:
        """Get logs from the coordinator agent."""
        return format_agent_logs(self.coordinator.get_succinct_logs())

    @property
    def agent_logs(self) -> list[dict[str, any]]:
        """Get logs from all web source agents."""
        all_logs = []
        for agent in self.agents:
            all_logs.extend(agent.logs)
        return all_logs

    def process_query(self, query: str) -> str:
        """Process a user query using all available web sources.
        
        Args:
            query: The user's information request
            
        Returns:
            str: The compiled response from all sources
        """
        # Reset all agents
        for agent in self.agents:
            agent.reset()
        self.content_extractor.reset()
        
        # Process the query
        return self.coordinator.run(task=query) 