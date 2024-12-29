# Multi-agentic RAG with Hugging Face Code Agents

This repository accompanies the blog post Multi-agentic RAG with Hugging Face Code Agents. 

Code agents work by executing Python code, make sure you run the notebook in an isolated environment and don't allow any unsafe import or function. Always supervise the system execution inspecting the logs, interrupt the execution if the system is trying to execute code that may pose risks.

As all Large Language Model based systems, this is also prone to hallucination. Always check the sources used to generate the answers and don't trust the generated answers blindly.

The default embedding model for similarity retrieval is all-mpnet-base-v2, check also the corresponding [model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
