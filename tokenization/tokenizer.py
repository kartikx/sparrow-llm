import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

def tokenize(input_text: str, tokenizer) -> torch.Tensor:
    # ideally this should get a tokenizer as a parameter.
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer(input_text, return_tensors="pt")["input_ids"]

"""
Input: [B, T, vocab_size]
Output: [B, 1]
"""
def sample(logits: torch.Tensor) -> torch.Tensor:
    # todo - we can skip softmax too.
    scores = F.softmax(logits, dim=-1) # [B, V]
    
    argmax_positions = torch.argmax(scores, dim=-1) # [B] 
    
    return argmax_positions
        
def detokenize(output_ids: torch.Tensor, tokenizer: any) -> str:
    return tokenizer.decode(output_ids)