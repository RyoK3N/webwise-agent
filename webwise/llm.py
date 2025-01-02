from functools import lru_cache
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast, StoppingCriteriaList
from typing import List, Dict, Optional
import logging

from webwise.utils import StopOnTokens

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerFast,
                 use_history_summary: bool = True,
                 cache_size: int = 1000,
                 device: Optional[str] = None,
                 **generate_kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.use_history_summary = use_history_summary
        self.generate_kwargs = generate_kwargs
        self.cache_size = cache_size
        
        # Set device (CPU, CUDA, or MPS)
        self.device = device or ('cuda' if torch.cuda.is_available() 
                               else 'mps' if torch.backends.mps.is_available() 
                               else 'cpu')
        self.model = self.model.to(self.device)
        
        # Enable model optimizations
        if self.device == 'cuda':
            self.model = torch.compile(self.model)  # PyTorch 2.0 compile for faster inference
        
        # Initialize cache
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize the LRU cache for message processing."""
        self.process_messages = lru_cache(maxsize=self.cache_size)(self._process_messages)

    @staticmethod
    def _hash_messages(messages: List[Dict[str, str]]) -> str:
        """Create a hash for messages to use as cache key."""
        return str(hash(str(messages)))

    def _process_messages(self, messages_hash: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process and format messages with history summarization."""
        try:
            if self.use_history_summary and len(messages) > 4:
                # Get past observations except errors
                past_observations = [
                    m['content'] for m in messages[:-1]
                    if m['role'] == 'tool-response' and "-> Error" not in m['content']
                ]
                # Clean output format
                past_observations = "\n".join(past_observations).replace('Print outputs:\n', '')
                return [
                    messages[0],
                    messages[1],
                    messages[-2],
                    {
                        "role": "user",
                        "content": f"Previous steps output:\n {past_observations} \n\n{messages[-1]['content']}"
                    }
                ]
            else:
                formatted_messages = messages.copy()
                for m in formatted_messages:
                    if m['role'] == 'tool-response':
                        m['role'] = 'user'
                return formatted_messages
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")
            return messages  # Fallback to original messages

    def get_stopping_criteria(self, stop_sequences: List[str]) -> StoppingCriteriaList:
        """Get stopping criteria for text generation."""
        return StoppingCriteriaList([StopOnTokens(stop_strings=stop_sequences, tokenizer=self.tokenizer)])

    @torch.inference_mode()  # Faster than no_grad for inference
    def __call__(self, 
                 messages: List[Dict[str, str]], 
                 stop_sequences: List[str] = '<end_action>',
                 max_retries: int = 3) -> str:
        """Generate response for given messages with error handling and retries."""
        for attempt in range(max_retries):
            try:
                # Process messages using cached function
                messages_hash = self._hash_messages(messages)
                formatted_messages = self.process_messages(messages_hash, messages)

                # Prepare inputs
                inputs = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)

                # Generate with stopping criteria
                stopping_criteria = self.get_stopping_criteria(stop_sequences)
                
                # Use batch processing if available
                if len(inputs.shape) == 1:
                    inputs = inputs.unsqueeze(0)
                
                with torch.inference_mode():
                    outputs = self.model.generate(
                        inputs,
                        stopping_criteria=stopping_criteria,
                        **self.generate_kwargs
                    )

                # Decode and return
                result = self.tokenizer.decode(
                    outputs[0][len(inputs[0]):],
                    skip_special_tokens=True
                )
                return result

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt < max_retries - 1:
                    logger.warning("CUDA OOM, clearing cache and retrying...")
                    continue
                raise

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error in generation attempt {attempt + 1}: {str(e)}")
                    continue
                logger.error(f"All generation attempts failed: {str(e)}")
                raise

    def clear_cache(self):
        """Clear the message processing cache."""
        self.process_messages.cache_clear() 