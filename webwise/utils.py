import torch
from transformers import StoppingCriteria, PreTrainedTokenizerFast


def format_agent_logs(logs: list[dict[str, any]]) -> dict[str, any]:
    """Format agent execution logs into a structured format."""
    task = logs[0]['task']
    execution_steps = []
    for i in range(1, len(logs)):
        step_data = logs[i]
        if 'observation' not in step_data:
            step_data['observation'] = None
        if 'llm_output' not in step_data:
            step_data['llm_output'] = None
        execution_steps.append({
            'llm_output': step_data['llm_output'],
            'observation': step_data['observation']
        })
    return {'task': task, 'execution_steps': execution_steps}


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    def __init__(self, stop_strings: list[str], tokenizer: PreTrainedTokenizerFast):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded = self.tokenizer.decode(input_ids[0])
        for stop_str in self.stop_strings:
            if stop_str in decoded.split('<|im_start|>assistant')[-1]:
                return True
        return False 