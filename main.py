from weight_loader import load_model

import argparse
import sys

def main(args: argparse.Namespace):
    load_model(args.model)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)