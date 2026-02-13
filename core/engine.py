import torch
from app.io_model import GenerateRequest
from core.cache import CacheManager
from core.weight_loader import load_model
from transformers import AutoTokenizer
from models.base import BaseLLMModel
from tokenization.tokenizer import tokenize, detokenize, sample

import logging

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, args):
        self.model_name = args.model
        self.dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: BaseLLMModel = load_model(self.model_name, self.dvc)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_num_seqs = getattr(args, "max_num_seqs", 4)
        self.max_seq_len = getattr(args, "max_seq_len", 32)

        kv_heads = self.model.config.get("num_key_value_heads") or self.model.config.get("num_attention_heads")    
        head_dim = self.model.config.get("head_dim") or self.model.config.get("hidden_size") // self.model.config.get("num_attention_heads")
        num_layers = self.model.config.get("num_hidden_layers") or None

        self.cache_manager = CacheManager(self.dvc,
                                          self.max_num_seqs,
                                          self.max_seq_len,
                                          num_layers,
                                          kv_heads,
                                          head_dim)
                                          
    def generate_output(self, req: GenerateRequest) -> str:
        # todo - this should be definitely be done before we get here.
        rid = self.cache_manager.alloc_one()
        if rid is None:
            raise ValueError("can not admit any more requests") 
        
        # todo - this should have already been tokenized. 
        # generate_output should work on a batch.
        input_ids = tokenize(req.prompt.strip(), self.tokenizer).to(self.dvc) # [B, T]

        logger.debug("input_ids: %s", {input_ids.shape})
        
        # prefill
        logits = self.model(input_ids, self.cache_manager, rid) # [B, T, V]

        for _ in range(req.max_tokens):
            next_token = sample(logits[:, -1, :]) # [B]
            next_input_ids = next_token.unsqueeze(1) # [B, T]
            logits = self.model(next_input_ids, self.cache_manager, rid)

            # extend original input_ids to store generated sequence.
            input_ids = torch.cat((input_ids, next_input_ids), dim=1)
            
        return detokenize(input_ids[0], self.tokenizer)

