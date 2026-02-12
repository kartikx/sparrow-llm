import torch
from io_model import GenerateRequest
from weight_loader import load_model
from transformers import AutoTokenizer
from tokenization.tokenizer import tokenize, detokenize, sample

import logging

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, args):
        self.model_name = args.model
        self.dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = load_model(self.model_name, self.dvc)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_output(self, req: GenerateRequest) -> str:
        kv_values = []
        input_ids = tokenize(req.prompt.strip(), self.tokenizer).to(self.dvc) # [B, T]

        logger.debug("input_ids: %s", {input_ids.shape})
        
        # prefill
        logits, kv_values = self.model(input_ids, kv_values) # [B, T, V]

        for _ in range(req.max_tokens):
            next_token = sample(logits[:, -1, :]) # [B]
            next_input_ids = next_token.unsqueeze(1) # [B, T]
            logits, kv_values = self.model(next_input_ids, kv_values)

            # extend original input_ids to store generated sequence.
            input_ids = torch.cat((input_ids, next_input_ids), dim=1)
            
        return detokenize(input_ids[0], self.tokenizer)

