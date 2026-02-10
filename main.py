from weight_loader import load_model
from tokenization.tokenizer import tokenize, detokenize, sample

import argparse
from transformers import AutoTokenizer
import torch

@torch.inference_mode()
def main(args: argparse.Namespace):
    dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading model')
    model = load_model(args.model, dvc)
    print('loaded model')
    
    input_text = "hey guys, it is me"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model) 
    
    kv_values = []
    input_ids = tokenize(input_text, tokenizer).to(dvc) # [B, T]
    print(f"input_ids: {input_ids.shape}")
    
    # prefill
    logits, kv_values = model(input_ids, kv_values) # [B, T, V]

    for _ in range(10):
        next_token = sample(logits[:, -1, :]) # [B]
        next_input_ids = next_token.unsqueeze(1) # [B, T]
        logits, kv_values = model(next_input_ids, kv_values)

        # extend original input_ids to store generated sequence.
        input_ids = torch.cat((input_ids, next_input_ids), dim=1)
        
        # only do for batch 0.
        print(detokenize(input_ids[0], tokenizer))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)