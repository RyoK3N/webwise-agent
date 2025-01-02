"""
System prompts and templates for the WebWise system.
"""

# Default system prompt for the base agent
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that provides accurate and informative responses 
based on web search results. Always cite your sources and provide balanced, factual information."""

# Template for web search queries
SEARCH_QUERY_TEMPLATE = """Please search for information about: {query}
Focus on finding reliable and up-to-date sources."""

# Template for processing web content
WEB_CONTENT_TEMPLATE = """Based on the following web content, please provide a comprehensive answer:

Content: {content}

Sources: {sources}

Please synthesize this information into a clear and accurate response."""

# Wikipedia-specific prompt
WIKI_PROMPT = """You are analyzing Wikipedia content. Please:
1. Focus on the main facts and key concepts
2. Maintain neutrality and accuracy
3. Include relevant citations
4. Highlight any areas of uncertainty or debate"""

# Custom website prompt template
CUSTOM_SITE_PROMPT = """Analyzing content from {domain}. Please:
1. Extract relevant information
2. Verify claims where possible
3. Note the source and publication date
4. Consider potential biases"""

# Response template
RESPONSE_TEMPLATE = """
Answer: {answer}

Sources:
{sources}

Confidence: {confidence}
""" 
