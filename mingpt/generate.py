#!/usr/bin/env python3
"""
Interactive question-answering script using a trained minGPT model.
Loads a saved model and allows you to type questions to get responses.
"""

import os
import sys
import torch
import argparse
import random

# Add the mingpt module to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mingpt.utils import CfgNode as CN
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.sentencepiece_tokenizer import load_tokenizer

def load_model(model_path, config_path=None):
    """
    Load a trained model from the given path.
    """
    # Load model configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        import json
        config_dict = json.load(f)
    
    # Create config object and properly handle nested structure
    config = CN()
    
    # Manually create nested structure
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # Create nested CfgNode for dictionaries
            nested_config = CN()
            nested_config.merge_from_dict(value)
            setattr(config, key, nested_config)
        else:
            # Set simple values directly
            setattr(config, key, value)
    
    # Ensure required model config values are present
    if not hasattr(config, 'model'):
        config.model = CN()
    if not hasattr(config.model, 'vocab_size') or config.model.vocab_size is None:
        config.model.vocab_size = 50257  # GPT-2 BPE vocab size
    if not hasattr(config.model, 'block_size') or config.model.block_size is None:
        config.model.block_size = config.data.block_size
    
    # Fix model config to avoid assertion error
    # Ensure all required attributes exist
    if not hasattr(config.model, 'n_layer'):
        config.model.n_layer = None
    if not hasattr(config.model, 'n_head'):
        config.model.n_head = None
    if not hasattr(config.model, 'n_embd'):
        config.model.n_embd = None
    
    # If model_type is specified, set individual parameters to None to avoid conflict
    if hasattr(config.model, 'model_type') and config.model.model_type is not None:
        # Set individual parameters to None so model_type takes precedence
        config.model.n_layer = None
        config.model.n_head = None
        config.model.n_embd = None
    
    # Create model
    print(f"Creating model with vocab_size={config.model.vocab_size}, block_size={config.model.block_size}")
    model = GPT(config.model)
    
    # Load trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print("Loading model weights...")
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    return model, config

def load_tokenizer_for_model(model_dir):
    """
    Load the appropriate tokenizer based on what's available in the model directory.
    """
    # Check for SentencePiece tokenizer first
    sp_model_path = os.path.join(model_dir, "tokenizer.model")
    if os.path.exists(sp_model_path):
        print("Loading SentencePiece tokenizer...")
        return load_tokenizer(sp_model_path)
    
    # Fall back to BPE tokenizer
    print("Loading BPE tokenizer...")
    return BPETokenizer()

def generate_response(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.95, top_k=10):
    """
    Generate a response to the given prompt.
    """
    # Tokenize the prompt
    if hasattr(tokenizer, 'encode'):
        # SentencePiece tokenizer
        encoded_prompt = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    else:
        # BPE tokenizer
        encoded_prompt = tokenizer(prompt)
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    encoded_prompt = encoded_prompt.to(device)
    
    # Generate response
    with torch.no_grad():
        generated_tokens = model.generate(
            encoded_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=top_k
        )
    
    # Decode the generated tokens
    if hasattr(tokenizer, 'decode'):
        # SentencePiece tokenizer
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
    else:
        # BPE tokenizer
        generated_text = tokenizer.decode(generated_tokens[0])
    
    # Extract only the new part (remove the original prompt)
    if hasattr(tokenizer, 'decode'):
        # SentencePiece tokenizer
        prompt_tokens = tokenizer.decode(encoded_prompt[0].tolist())
    else:
        # BPE tokenizer
        prompt_tokens = tokenizer.decode(encoded_prompt[0])
    
    response = generated_text[len(prompt_tokens):]
    
    # Stop at <eot> token if found
    
    
    return response.strip()

def main():
    # Set random seed for reproducibility (different each run)
    random.seed()
    torch.manual_seed(random.randint(1, 1000000))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random.randint(1, 1000000))
    
    parser = argparse.ArgumentParser(description='Interactive question-answering with minGPT')
    parser.add_argument('--model_path', type=str, default='./out/chargpt/model.pt',
                       help='Path to the saved model file')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to the model config file (optional, will be inferred from model_path)')
    parser.add_argument('--max_tokens', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Top-k sampling parameter')
    
    args = parser.parse_args()
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load model
        print(f"Loading model from {args.model_path}...")
        model, config = load_model(args.model_path, args.config_path)
        model = model.to(device)
        print("Model loaded successfully!")
        
        # Load appropriate tokenizer
        model_dir = os.path.dirname(args.model_path)
        tokenizer = load_tokenizer_for_model(model_dir)
        
        print("\n" + "="*60)
        print("Interactive Chat with minGPT")
        print("="*60)
        print("Start typing your message. Press Enter to send.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'help' for available commands.")
        print("-" * 60)
        
        # Start the conversation with <user> token
        conversation = "<user>"
        torch.set_default_dtype(torch.bfloat16)
        while True:
            try:
                # Get user input (single line)
                print("\nYou: ", end="", flush=True)
                user_input = input().strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help message")
                    print("  quit/exit/q - Exit the program")
                    print("  clear - Clear the screen")
                    print("  reset - Reset the conversation")
                    print("\nJust type your message and press Enter to send!")
                    continue
                
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                if user_input.lower() == 'reset':
                    conversation = "<user>"
                    print("Conversation reset!")
                    continue
                
                if not user_input:
                    continue
                
                # Add user input to conversation
                conversation += f" {user_input}<ai>"
                
                print("", end="", flush=True)
                
                # Generate response
                response = generate_response(
                    model, 
                    tokenizer, 
                    conversation, 
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                
                print(response)
                
                # Add AI response to conversation for context
                conversation += f" {response}"
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. A trained model file (model.pt)")
        print("2. A corresponding config file (config.json)")
        print("3. Both files should be in the same directory")
        print("4. For SentencePiece models: a tokenizer.model file")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
