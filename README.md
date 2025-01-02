# WebWise Agent

An intelligent multi-agent system for web-based information retrieval and analysis, built with state-of-the-art language models and efficient information processing.

## Features

- **Multi-Source Information Retrieval**
  - Wikipedia integration with smart content extraction
  - Custom website scraping and analysis
  - Extensible architecture for additional sources

- **Advanced Processing**
  - Parallel content processing
  - Smart caching system (memory and disk-based)
  - Efficient embedding computation
  - Batch processing capabilities

- **Performance Optimizations**
  - GPU acceleration support (CUDA)
  - Apple Silicon support (MPS)
  - PyTorch 2.0 optimizations
  - Parallel processing for large content
  - Caching for repeated queries

- **Intelligent Agents**
  - Coordinator Agent for query orchestration
  - Specialized Wiki Search Agent
  - Content Extraction Agent
  - Custom Website Agent

## Installation

```bash
# Basic installation
pip install git+https://github.com/RyoK3N/webwise-agent.git

# With CUDA support
pip install git+https://github.com/RyoK3N/webwise-agent.git#egg=webwise-agent[cuda]

# With Apple Silicon optimization
pip install git+https://github.com/RyoK3N/webwise-agent.git#egg=webwise-agent[mps]

# With development tools
pip install git+https://github.com/RyoK3N/webwise-agent.git#egg=webwise-agent[dev]
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from webwise.system import WebWiseSystem
from webwise.config import WebSearchConfig, WebSourceConfig, WebSourceType

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Configure system
system = WebWiseSystem(
    model=model,
    tokenizer=tokenizer,
    search_cfg=WebSearchConfig(
        sources=[
            # Wikipedia source
            WebSourceConfig(
                source_type=WebSourceType.WIKI
            ),
            # Custom website source
            WebSourceConfig(
                source_type=WebSourceType.CUSTOM,
                base_url="https://example.com",
                system_prompt="Custom prompt for this source..."
            )
        ]
    )
)

# Process a query
result = system.process_query("What is quantum computing?")
print(result)
```

## Architecture

The system consists of several key components:

1. **WebWiseSystem**: Main coordinator that manages all agents and processes
2. **LLMEngine**: Optimized language model interface with caching
3. **ContentExtractor**: Parallel content processing and embedding computation
4. **Specialized Agents**:
   - WikiAgent: Wikipedia search and content extraction
   - CustomWebAgent: Custom website scraping and analysis
   - More agents can be added by implementing the WebSourceAgent interface

## Performance Features

- **Caching System**
  - LRU caching for message processing
  - Disk-based caching for embeddings
  - Memory caching for frequent operations

- **Parallel Processing**
  - Multi-threaded content processing
  - Batch embedding computation
  - Concurrent web requests

- **Hardware Optimization**
  - Automatic device selection (CUDA/MPS/CPU)
  - PyTorch 2.0 compilation for CUDA
  - Optimized memory usage

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black webwise/
isort webwise/

# Type checking
mypy webwise/

# Lint
ruff check webwise/
```

## Warning

This system executes Python code and makes web requests. Always:
- Run in an isolated environment
- Monitor system execution through logs
- Review and validate generated content
- Be cautious with API keys and sensitive data

## License

MIT License - See LICENSE file for details.

## Author

Reiyo (reiyo1113@gmail.com)
