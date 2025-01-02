import wikipedia
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import tool, ReactCodeAgent, Tool
from bs4 import BeautifulSoup
import requests
from abc import ABC, abstractmethod
import concurrent.futures
from functools import lru_cache
import logging
from typing import List, Dict, Optional, Any
import diskcache

from webwise.config import ContentExtractionConfig, WebSearchConfig
from webwise.llm import LLMEngine
from webwise.utils import format_agent_logs

logger = logging.getLogger(__name__)

class ContentExtractor:
    def __init__(self,
                 llm_engine: LLMEngine,
                 embeddings: HuggingFaceEmbeddings,
                 system_prompt: str,
                 extraction_cfg: ContentExtractionConfig = ContentExtractionConfig(),
                 cache_dir: Optional[str] = ".cache/content_extractor"):
        self.embeddings = embeddings
        self.extraction_cfg = extraction_cfg
        self.content_processor = ReactCodeAgent(
            system_prompt=system_prompt,
            llm_engine=llm_engine,
            max_iterations=extraction_cfg.max_retries,
            tools=[self.content_extraction_tool()]
        )
        self.current_content = None
        self.vector_store = None
        self.logs = []
        
        # Initialize caches
        self.cache = diskcache.Cache(cache_dir) if cache_dir else None
        self._initialize_caches()

    def _initialize_caches(self):
        """Initialize LRU caches for expensive operations."""
        self.compute_embeddings = lru_cache(maxsize=1000)(self._compute_embeddings)
        self.process_content = lru_cache(maxsize=1000)(self._process_content)

    def _compute_embeddings(self, content: str) -> FAISS:
        """Compute embeddings for content with caching."""
        if self.cache:
            cache_key = f"embeddings_{hash(content)}"
            if cache_key in self.cache:
                return self.cache[cache_key]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.extraction_cfg.chunk_size,
            chunk_overlap=self.extraction_cfg.chunk_overlap,
            separators=self.extraction_cfg.text_separators
        )
        
        # Process content in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            content_chunks = splitter.create_documents(texts=[content])
            chunk_size = 10  # Process embeddings in batches
            chunks = [content_chunks[i:i + chunk_size] 
                     for i in range(0, len(content_chunks), chunk_size)]
            
            # Compute embeddings in parallel
            vector_stores = list(executor.map(
                lambda x: FAISS.from_documents(x, self.embeddings),
                chunks
            ))
            
            # Merge vector stores
            vector_store = vector_stores[0]
            for vs in vector_stores[1:]:
                vector_store.merge_from(vs)

        if self.cache:
            self.cache[cache_key] = vector_store
        return vector_store

    def _process_content(self, query: str, content_id: str) -> str:
        """Process content with caching."""
        cache_key = f"content_{hash(f'{query}_{content_id}')}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            self.current_content = wikipedia.page(content_id, auto_suggest=False)
            self.vector_store = self.compute_embeddings(self.current_content.content)
            
            task = f"""Find information about: "{query}" in "{self.current_content.title}"."""
            output = self.content_processor.run(task)
            output = f"Information from '{self.current_content.title}' about '{query}':\n" + output
            
            if self.cache:
                self.cache[cache_key] = output
            return output
            
        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            return f"Error processing content: {str(e)}"
        finally:
            self.reset_content()

    def content_extraction_tool(self) -> Tool:
        @tool
        def extract_content(query: str, content_id: str) -> str:
            """Extract relevant information from the content based on the query.
            Args:
                query: The information to look for.
                content_id: Identifier for the content source.
            Returns:
                str: The extracted information.
            """
            output = self.process_content(query, content_id)
            self.logs.append(format_agent_logs(self.content_processor.get_succinct_logs()))
            return output

        return extract_content

    def reset_content(self):
        """Reset current content and vector store."""
        self.current_content = None
        self.vector_store = None

    def reset(self) -> None:
        """Reset logs and clear caches."""
        self.logs = []
        if self.cache:
            self.cache.clear()
        self.compute_embeddings.cache_clear()
        self.process_content.cache_clear()


class WebSourceAgent(ABC):
    def __init__(self,
                 llm_engine: LLMEngine,
                 content_extractor: ContentExtractor,
                 system_prompt: str,
                 search_cfg: WebSearchConfig = WebSearchConfig()):
        self.search_agent = ReactCodeAgent(system_prompt=system_prompt,
                                         llm_engine=llm_engine,
                                         tools=[self.search_tool(),
                                               content_extractor.content_extraction_tool()])
        self.search_cfg = search_cfg
        self.logs = []

    @abstractmethod
    def search_tool(self) -> Tool:
        """Implement the specific search functionality for the web source"""
        pass

    def reset(self) -> None:
        self.logs = []

    def as_tool(self) -> Tool:
        @tool
        def web_source_agent(query: str) -> str:
            """Search and extract information from a web source.
            Args:
                query: The information to search for.
            Returns:
                str: The retrieved information.
            """
            output = self.search_agent.run(query)
            self.logs.append(format_agent_logs(self.search_agent.get_succinct_logs()))
            return output

        return web_source_agent


class WikiAgent(WebSourceAgent):
    def search_tool(self) -> Tool:
        @tool
        def search_wiki(query: str) -> str:
            """Search Wikipedia for relevant pages.
            Args:
                query: The search query.
            Returns:
                str: Found pages and their summaries.
            """
            pages = wikipedia.search(query, results=self.search_cfg.results_per_source)
            results = f"Wikipedia pages for '{query}':\n"
            for p in pages:
                try:
                    wiki_page = wikipedia.page(p, auto_suggest=False)
                    results += f"Page: {wiki_page.title}\nSummary: {wiki_page.summary[:self.search_cfg.max_content_length]}\n"
                except wikipedia.exceptions.PageError:
                    continue
            return results


class CustomWebAgent(WebSourceAgent):
    def __init__(self, 
                 llm_engine: LLMEngine,
                 content_extractor: ContentExtractor,
                 system_prompt: str,
                 base_url: str,
                 search_cfg: WebSearchConfig = WebSearchConfig()):
        super().__init__(llm_engine, content_extractor, system_prompt, search_cfg)
        self.base_url = base_url

    def search_tool(self) -> Tool:
        @tool
        def search_website(query: str) -> str:
            """Search a specific website for relevant content.
            Args:
                query: The search query.
            Returns:
                str: Retrieved content.
            """
            try:
                response = requests.get(f"{self.base_url}/search?q={query}")
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return f"Content for '{query}':\n{text[:self.search_cfg.max_content_length]}"
            except Exception as e:
                return f"Error retrieving content: {str(e)}"

        return search_website 