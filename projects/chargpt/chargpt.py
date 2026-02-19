"""
Trains a language model using SentencePiece tokenization.
"""

import os
import sys
import pickle
import random
import math

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset

try:
    from mingpt.utils import set_seed, setup_logging, CfgNode as CN
    from mingpt.model import GPT
    from mingpt.trainer import Trainer
    from mingpt.sentencepiece_tokenizer import SentencePieceDataset, save_tokenizer
except ImportError:
    # Add the root directory (where mingpt module is located) to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)  # projects/
    root_dir = os.path.dirname(root_dir)     # root directory with mingpt/
    sys.path.insert(0, root_dir)
    from mingpt.utils import set_seed, setup_logging, CfgNode as CN
    from mingpt.model import GPT
    from mingpt.trainer import Trainer
    from mingpt.sentencepiece_tokenizer import SentencePieceDataset, save_tokenizer

# -----------------------------------------------------------------------------

class TokenChunkDataset(Dataset):
    """Simple contiguous token-window dataset usable by multiprocessing workers."""
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_config():
    
    C = CN()

    # system
    C.system = CN()
    random.seed()
    
    C.system.seed = random.randint(1, 1000000)
    C.system.work_dir = './out/chargpt'

    # data
    C.data = SentencePieceDataset.get_default_config()
    # Add option to show training data sample
    C.data.show_sample = False  # Set to True to display a sample of training data
    C.data.sample_size = 10000   # Number of characters to show in the sample
    # Path to save/load tokenized data for faster subsequent runs
    C.data.tokenized_data_path = './out/chargpt/tokenized_data.pkl'
    # For HuggingFace datasets, you can specify which column contains the text
    # C.data.text_column = "text"  # Uncomment and set if auto-detection fails

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-micro'
    
    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 1e-4  # Increased learning rate for better training
    C.trainer.max_iters = 3500000  # Set a reasonable max iterations
    # Reduce validation overhead
    C.trainer.eval_interval = 500
    C.trainer.eval_iters = 500
    
    # Learning rate schedule parameters
    C.trainer.warmup_iters = 10000  # Number of iterations for warm-up
    C.trainer.lr_decay_iters = 3061204  # Number of iterations for linear decay
    C.trainer.min_lr = 5e-6  # Minimum learning rate after decay
    
    C.attn_band_width = 4
    C.model.use_dape = True
    #C.model.dape_mode = 'replace'
    
    return C

# -----------------------------------------------------------------------------

def load_chatgpt_prompts_dataset(cache_dir='./cache'):
    """
    Load the ChatGPT prompts dataset and return a single concatenated text for training.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'chatgpt_prompts_processed.pkl')
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        print("Loading cached processed data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Loading ChatGPT prompts dataset from Hugging Face...")
    
    ds = load_dataset("alespalla/chatbot_instruction_prompts")
    
    # Create a single text string for training
    all_text = ""
    
    for split in ['train', 'validation', 'test']:
        if split in ds:
            print(f"Processing {split} split...")
            for item in ds[split]:
                human_prompt = item.get('prompt', '')
                chatgpt_response = item.get('response', '')
                
                # Skip empty prompts or responses
                if not human_prompt.strip() or not chatgpt_response.strip():
                    continue
                
                # Format as a conversation
                conversation = f"{human_prompt}. {chatgpt_response}"
                all_text += conversation
    
    print(f"Processed {len(all_text)} characters of training data")
    
    # Cache the processed data
    print("Caching processed data...")
    with open(cache_file, 'wb') as f:
        pickle.dump(all_text, f)
    
    return all_text


def show_training_data_sample(data, sample_size=1000):
    """
    Display a sample of the training data for inspection.
    Works with both text strings and HuggingFace datasets.
    """
    print("\n" + "="*80)
    print("TRAINING DATA SAMPLE")
    print("="*80)
    
    # Handle HuggingFace Dataset or DatasetDict
    if hasattr(data, 'column_names') or hasattr(data, 'keys'):
        print("Dataset type: HuggingFace Dataset")
        
        # If it's a DatasetDict, choose a split (prefer 'train')
        dataset = None
        if hasattr(data, 'keys') and callable(getattr(data, 'keys')):
            available_splits = list(data.keys())
            print(f"Available splits: {available_splits}")
            chosen_split = 'train' if 'train' in available_splits else available_splits[0]
            dataset = data[chosen_split]
            print(f"Using '{chosen_split}' split with {len(dataset)} examples")
        else:
            dataset = data
            print(f"Single dataset with {len(dataset)} examples")
        
        # Show columns for the chosen dataset
        try:
            print(f"Dataset columns: {dataset.column_names}")
        except Exception:
            pass
        
        # Show first few examples
        print(f"\nShowing first {min(5, len(dataset))} examples:")
        print("-"*80)
        for i in range(min(5, len(dataset))):
            example = dataset[i]
            print(f"Example {i+1}:")
            for col, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {col}: {value[:100]}...")
                else:
                    print(f"  {col}: {value}")
            print()
        
        # Show statistics
        print("-"*80)
        print("DATASET STATISTICS:")
        print(f"Total examples: {len(dataset):,}")
        
        # Count total characters across all examples
        total_chars = 0
        unique_chars = set()
        for i in range(min(1000, len(dataset))):  # Sample first 1000 examples
            example = dataset[i]
            for col, value in example.items():
                if isinstance(value, str):
                    total_chars += len(value)
                    unique_chars.update(value)
        
        print(f"Total characters (sampled): {total_chars:,}")
        print(f"Unique characters (sampled): {len(unique_chars)}")
        
    else:
        # Text string dataset
        print("Dataset type: Text String")
        print(f"Showing first {sample_size} characters of training data:")
        print("-"*80)
        
        # Show the sample
        sample = data[:sample_size]
        
        print("\n" + "-"*80)
        print("Sample as readable text:")
        print("-"*80)
        print(sample)
        
        # Show some statistics
        print("\n" + "-"*80)
        print("DATA STATISTICS:")
        print(f"Total characters: {len(data):,}")
        print(f"Unique characters: {len(set(data))}")
        print(f"Sample size shown: {len(sample)}")
        
        # Show character frequency in sample
        char_freq = {}
        for char in sample:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        print(f"\nMost common characters in sample:")
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        for char, freq in sorted_chars[:10]:
            char_repr = repr(char)[1:-1]  # Remove quotes from repr
            print(f"  '{char_repr}': {freq}")
    
    print("="*80 + "\n")


def get_lr_scheduler(optimizer, config):
    """
    Create a learning rate scheduler with warm-up and linear decay.
    
    Args:
        optimizer: The optimizer to schedule
        config: Training configuration containing lr schedule parameters
    
    Returns:
        A function that updates the learning rate based on current iteration
    """
    def update_lr(iter_num):
        # Warm-up phase: linear increase from 0 to target learning rate
        if iter_num < config.warmup_iters:
            lr = config.learning_rate * iter_num / config.warmup_iters
        # Decay phase: linear decrease from target learning rate to min_lr
        elif iter_num < config.warmup_iters + config.lr_decay_iters:
            decay_ratio = (iter_num - config.warmup_iters) / config.lr_decay_iters
            lr = config.learning_rate - (config.learning_rate - config.min_lr) * decay_ratio
        # Constant phase: maintain min_lr
        else:
            lr = config.min_lr
        
        # Update learning rate for all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    return update_lr


if __name__ == '__main__':
    # get default config and overrides from the command line, if any
    
    torch.set_default_dtype(torch.bfloat16)

    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    # Choose your dataset here:
    use_huggingface_dataset = False  # Set to False to use the old ChatGPT prompts dataset
    
    if use_huggingface_dataset:
        print("Loading HuggingFace dataset: AI-companionship/INTIMA")
        text = load_dataset("AI-companionship/INTIMA")
        # You can also specify which text column to use if auto-detection fails
        # config.data.text_column = "text"  # Uncomment and set if needed
    else:
        print("Loading ChatGPT prompts dataset")
        text = load_dataset("alespalla/chatbot_instruction_prompts")
        #text = load_chatgpt_prompts_dataset()
    
    # Show training data sample if requested
    if config.data.show_sample:
        show_training_data_sample(text, config.data.sample_size)
    
    # Create SentencePiece dataset (this will train the tokenizer if needed)
    # The dataset will automatically save tokenized data to config.data.tokenized_data_path
    # and load it on subsequent runs for faster startup
    print("Creating SentencePiece dataset...")
    sp_dataset = SentencePieceDataset(config.data, text)

    # Build train/val split from the contiguous token stream

    all_tokens = sp_dataset.encoded_data
    val_ratio = getattr(config.trainer, 'val_split', 0.1)
    split_idx = int(len(all_tokens) * (1.0 - val_ratio))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    train_dataset = TokenChunkDataset(train_tokens, sp_dataset.get_block_size())
    val_dataset = TokenChunkDataset(val_tokens, sp_dataset.get_block_size())

    # construct the model
    config.model.vocab_size = sp_dataset.get_vocab_size()
    config.model.block_size = sp_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset, val_dataset=val_dataset)
    
    # Track training progress
    total_samples = len(train_dataset)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate iterations needed to reach target tokens-per-parameter
    tokens_per_iter = trainer.config.batch_size * config.model.block_size
    target_tokens_per_param = 20
    required_iters_for_target_tpp = math.ceil((target_tokens_per_param * total_params) / tokens_per_iter)
    print(f"To reach ~{target_tokens_per_param} tokens/param: need ≈ {required_iters_for_target_tpp:,} iterations (tokens/iter={tokens_per_iter:,}, params={total_params:,})")
    
    # iteration callback
    def batch_end_callback(trainer):
        # Create learning rate scheduler if not already created
        if not hasattr(trainer, 'lr_scheduler'):
            trainer.lr_scheduler = get_lr_scheduler(trainer.optimizer, config.trainer)
        
        # Update learning rate based on current iteration
        current_lr = trainer.lr_scheduler(trainer.iter_num)
        
        # Calculate percentage towards max_iters
        progress_percent = (trainer.iter_num / trainer.config.max_iters) * 100
        
        # Calculate input tokens per parameter
        tokens_processed = trainer.iter_num * trainer.config.batch_size * config.model.block_size
        tokens_per_param = tokens_processed / total_params
        
        if trainer.iter_num % getattr(config.trainer, 'eval_interval', 50) == 0:
            val_loss = trainer.estimate_loss()
            if val_loss is not None:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f} | val loss {val_loss:.5f} | lr {current_lr:.2e} | Progress: {progress_percent:.1f}% | Tokens/param: {tokens_per_param:.2f}")
            else:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f} | lr {current_lr:.2e} | Progress: {progress_percent:.1f}% | Tokens/param: {tokens_per_param:.2f}")

        if trainer.iter_num % 5000 == 0:
            # evaluate both the train and test score
            if (trainer.iter_num > 0):
                model.eval()
                with torch.no_grad():
                    # sample from the model...
                    context = "The "
                    x = torch.tensor(sp_dataset.encode(context), dtype=torch.long).unsqueeze(0).to(trainer.device)
                    y = model.generate(x, 200, temperature=0.8, do_sample=True, top_k=40)[0]
                    completion = sp_dataset.decode(y.tolist())
                    print(f"\n--- GENERATION SAMPLE (iter {trainer.iter_num}) ---")
                    #print(f"Context: {context}")
                    print(f"Generated: {completion}")
                    print("--- END GENERATION ---\n")
                    
                    print("saving model")
                    ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    
                    # Save the tokenizer
                    tokenizer_path = os.path.join(config.system.work_dir, "tokenizer.model")
                    save_tokenizer(sp_dataset.tokenizer, tokenizer_path)
                    
                    # Save updated config with vocab_size and block_size
                    config_path = os.path.join(config.system.work_dir, "config.json")
                    with open(config_path, 'w') as f:
                        import json
                        json.dump(config.to_dict(), f, indent=4)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
