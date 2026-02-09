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
    
    for _ in range(20):
        input_ids = tokenize(input_text, tokenizer).to(dvc)
    
        logits = model(input_ids)
    
        output_ids = sample(logits)
    
        next_token = detokenize(output_ids, tokenizer)
        
        input_text += next_token

        print(input_text)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)