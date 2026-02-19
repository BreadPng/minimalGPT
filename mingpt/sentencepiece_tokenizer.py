"""
SentencePiece tokenizer implementation for minGPT.
This provides a more flexible tokenization approach than the fixed GPT-2 BPE tokenizer.
"""

import os
import pickle
import torch
from torch.utils.data import Dataset
import sentencepiece as spm


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper for minGPT"""
    
    def __init__(self, model_path=None, vocab_size=8000, character_coverage=0.9995):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.sp = None
        
        if model_path and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            print(f"Loaded SentencePiece model from {model_path}")
            print(f"Vocabulary size: {self.sp.get_piece_size()}")
        else:
            print("No existing model found. Will train a new one.")
    
    def train(self, text_file, model_prefix="sp_model"):
        """Train a new SentencePiece model on the given text file"""
        print(f"Training SentencePiece model on {text_file}...")
        
        # SentencePiece training parameters
        train_args = [
            f'--input={text_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={self.vocab_size}',
            f'--character_coverage={self.character_coverage}',
            '--model_type=bpe',
            '--pad_id=0',
            '--unk_id=1',
            '--bos_id=2',
            '--eos_id=3',
            '--pad_piece=<pad>',
            '--unk_piece=<unk>',
            '--bos_piece=<s>',
            '--eos_piece=</s>',
            '--user_defined_symbols=<user>,<ai>,<eot>',  # Special tokens for conversation
            '--split_by_unicode_script=false',
            '--split_by_number=true',
            '--split_by_whitespace=true',
            '--treat_whitespace_as_suffix=false',
            '--allow_whitespace_only_pieces=true',
            '--split_digits=true',
            '--byte_fallback=false',
            '--unk_surface=<unk>',
            '--minloglevel=1'
        ]
        
        spm.SentencePieceTrainer.train(' '.join(train_args))
        
        # Load the trained model
        self.model_path = f"{model_prefix}.model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)
        
        print(f"Training completed. Model saved to {self.model_path}")
        print(f"Vocabulary size: {self.sp.get_piece_size()}")
        
        return self.model_path
    
    def encode(self, text):
        """Encode text to token IDs"""
        if self.sp is None:
            raise ValueError("SentencePiece model not loaded. Call train() first.")
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids):
        """Decode token IDs back to text"""
        if self.sp is None:
            raise ValueError("SentencePiece model not loaded. Call train() first.")
        return self.sp.decode_ids(ids)
    
    def encode_as_pieces(self, text):
        """Encode text to token pieces (strings)"""
        if self.sp is None:
            raise ValueError("SentencePiece model not loaded. Call train() first.")
        return self.sp.encode_as_pieces(text)
    
    def get_vocab_size(self):
        """Get the vocabulary size"""
        if self.sp is None:
            return self.vocab_size  # Return default if not trained yet
        return self.sp.get_piece_size()
    
    def __call__(self, text, return_tensors='pt'):
        """PyTorch-compatible interface"""
        assert return_tensors == 'pt'
        assert isinstance(text, str)
        
        # Encode and create batch dimension
        ids = [self.encode(text)]
        return torch.tensor(ids, dtype=torch.long)


class SentencePieceDataset(Dataset):
    """Dataset for SentencePiece tokenized text - properly handles HuggingFace datasets"""
    
    @staticmethod
    def get_default_config():
        from mingpt.utils import CfgNode as CN
        C = CN()
        C.block_size = 128
        C.vocab_size = 10000
        C.character_coverage = 0.9995
        C.model_prefix = "sp_model"
        C.tokenized_data_path = None  # Path to save/load tokenized data
        C.text_column = None  # Column name for text in HuggingFace datasets
        # Optional: combine multiple text columns (e.g., ['prompt','response'])
        C.combine_columns = ['prompt', 'response']
        C.combine_separator = " "
        return C
    
    def __init__(self, config, data, tokenizer=None):
        self.config = config
        self.data = data
        self.combine_columns = list(getattr(config, 'combine_columns', []) or [])
        self.combine_separator = getattr(config, 'combine_separator', ' ')
        
        # Initialize or use provided tokenizer
        if tokenizer is None:
            self.tokenizer = SentencePieceTokenizer(
                vocab_size=config.vocab_size,
                character_coverage=config.character_coverage
            )
        else:
            self.tokenizer = tokenizer
        
        # Handle different data types
        if self._is_huggingface_dataset(data):
            self._setup_huggingface_dataset(data)
        else:
            self._setup_text_dataset(data)
        
        # Train tokenizer if needed
        if self.tokenizer.sp is None:
            self._train_tokenizer()
        
        # Tokenize the dataset
        self._tokenize_dataset()
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"Final vocabulary size: {self.vocab_size}")
        print(f"Dataset length: {len(self)}")
    
    def _is_huggingface_dataset(self, data):
        """Check if data is a HuggingFace dataset"""
        return hasattr(data, 'column_names') and hasattr(data, '__getitem__')
    
    def _setup_huggingface_dataset(self, data):
        """Setup for HuggingFace dataset"""
        print("Detected HuggingFace dataset")
        
        # Determine which split to use
        if hasattr(data, 'keys') and 'train' in data.keys():
            # DatasetDict with multiple splits
            self.dataset = data['train']
            print(f"Using 'train' split with {len(self.dataset)} examples")
        else:
            # Single dataset
            self.dataset = data
            print(f"Single dataset with {len(self.dataset)} examples")
        
        # Determine text extraction strategy: combine vs single column
        self.text_column = None
        if self.combine_columns:
            missing = [c for c in self.combine_columns if c not in self.dataset.column_names]
            if missing:
                raise ValueError(f"combine_columns contains columns not in dataset: {missing}. Available: {self.dataset.column_names}")
            print(f"Combining columns for text: {self.combine_columns} (sep='{self.combine_separator}')")
        else:
            if self.config.text_column and self.config.text_column in self.dataset.column_names:
                self.text_column = self.config.text_column
            else:
                # Auto-detect a reasonable text column
                text_columns = ['text', 'content', 'message', 'prompt', 'response', 'conversation']
                for col in text_columns:
                    if col in self.dataset.column_names:
                        self.text_column = col
                        break
                if self.text_column is None:
                    # Use first column that looks like text
                    for col in self.dataset.column_names:
                        if len(self.dataset) > 0 and self.dataset[0][col] and isinstance(self.dataset[0][col], str):
                            self.text_column = col
                            break
                if self.text_column is None:
                    raise ValueError(f"Could not identify text column. Available columns: {self.dataset.column_names}")
            print(f"Using text column: '{self.text_column}'")
        
        # Calculate appropriate vocab size based on data
        #self._calculate_vocab_size()
    
    def _setup_text_dataset(self, data):
        """Setup for text string dataset"""
        print("Detected text string dataset")
        self.dataset = None
        self.text_column = None
        self.text_data = str(data)
        print(f"Text length: {len(self.text_data)} characters")
        
        # Calculate appropriate vocab size
        unique_chars = len(set(self.text_data))
        suggested_vocab = min(unique_chars * 2, 8000)
        if suggested_vocab < self.tokenizer.vocab_size:
            print(f"Adjusting vocab size from {self.tokenizer.vocab_size} to {suggested_vocab}")
            self.tokenizer.vocab_size = suggested_vocab
    
    def _calculate_vocab_size(self):
        """Calculate appropriate vocabulary size for the dataset"""
        if self.dataset is None:
            return
        
        # Sample some examples to estimate vocabulary needs
        sample_size = min(1000, len(self.dataset))
        sample_texts = []
        
        for i in range(sample_size):
            text = self._extract_text(self.dataset[i])
            if text.strip():
                sample_texts.append(text)
        
        if sample_texts:
            combined_sample = ' '.join(sample_texts)
            unique_chars = len(set(combined_sample))
            suggested_vocab = min(unique_chars * 2, 8000)
            
            if suggested_vocab < self.tokenizer.vocab_size:
                print(f"Adjusting vocab size from {self.tokenizer.vocab_size} to {suggested_vocab}")
                self.tokenizer.vocab_size = suggested_vocab
    
    def _train_tokenizer(self):
        """Train the SentencePiece tokenizer"""
        # Check if we have a saved model to load
        model_path = f"{self.config.model_prefix}.model"
        if os.path.exists(model_path):
            print(f"Loading existing SentencePiece model from {model_path}")
            self.tokenizer = SentencePieceTokenizer(model_path=model_path)
            return
        
        # Create training file
        temp_file = "temp_training_data.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            if self.dataset is not None:
                # HuggingFace dataset
                for i, example in enumerate(self.dataset):
                    text = self._extract_text(example)
                    if text.strip():
                        f.write(text + '\n')
                    
                    # Limit training examples to avoid memory issues
                    if i >= 10000:
                        print(f"Limited training to first {i+1} examples")
                        break
            else:
                # Text dataset
                f.write(self.text_data)
        
        # Train the tokenizer
        model_path = self.tokenizer.train(temp_file, self.config.model_prefix)
        
        # Clean up temp file
        os.remove(temp_file)
    
    def _tokenize_dataset(self):
        """Tokenize the entire dataset"""
        if self.config.tokenized_data_path and os.path.exists(self.config.tokenized_data_path):
            print(f"Loading pre-tokenized data from {self.config.tokenized_data_path}")
            self._load_tokenized_data(self.config.tokenized_data_path)
        else:
            print("Tokenizing dataset...")
            
            if self.dataset is not None:
                # HuggingFace dataset
                all_tokens = []
                for i, example in enumerate(self.dataset):
                    text = self._extract_text(example)
                    if text.strip():
                        tokens = self.tokenizer.encode(text)
                        all_tokens.extend(tokens)
                    # Progress indicator
                    if i % 1000 == 0 and i > 0:
                        print(f"Tokenized {i} examples...")
                self.encoded_data = all_tokens
            else:
                # Text dataset
                self.encoded_data = self.tokenizer.encode(self.text_data)
            
            print(f"Dataset tokenized. Total tokens: {len(self.encoded_data)}")
            
            # Save tokenized data if path is specified
            if self.config.tokenized_data_path:
                print(f"Saving tokenized data to {self.config.tokenized_data_path}")
                self._save_tokenized_data(self.config.tokenized_data_path)
    
    def _save_tokenized_data(self, filepath):
        """Save tokenized data to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_to_save = {
            'encoded_data': self.encoded_data,
            'vocab_size': self.tokenizer.get_vocab_size(),
            'text_length': len(self.text_data) if hasattr(self, 'text_data') else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
    
    def _load_tokenized_data(self, filepath):
        """Load tokenized data from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.encoded_data = data['encoded_data']
        print(f"Loaded {len(self.encoded_data)} tokens from {filepath}")
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        return self.config.block_size
    
    def __len__(self):
        length = len(self.encoded_data) - self.config.block_size
        return max(0, length)  # Ensure we never return negative values
    
    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) tokens from the data
        chunk = self.encoded_data[idx:idx + self.config.block_size + 1]
        # Return as tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text):
        """Encode text to token indices"""
        return self.tokenizer.encode(text)
    
    def decode(self, indices):
        """Decode token indices back to text"""
        return self.tokenizer.decode(indices)

    def _extract_text(self, example):
        """Extract text from an example according to config: combine or single column."""
        if self.dataset is not None:
            if self.combine_columns:
                parts = []
                for col in self.combine_columns:
                    val = example.get(col, '')
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                return self.combine_separator.join(parts)
            elif self.text_column is not None and self.text_column in example:
                val = example[self.text_column]
                return str(val) if isinstance(val, str) else ''
        return ''


def save_tokenizer(tokenizer, save_path):
    """Save a trained tokenizer"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the SentencePiece model
    if tokenizer.model_path:
        import shutil
        shutil.copy2(tokenizer.model_path, save_path)
    
    # Save metadata
    metadata = {
        'vocab_size': tokenizer.vocab_size,
        'character_coverage': tokenizer.character_coverage,
        'model_path': tokenizer.model_path
    }
    
    metadata_path = save_path.replace('.model', '_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


def load_tokenizer(load_path):
    """Load a trained tokenizer"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Tokenizer model not found at {load_path}")
    
    # Load metadata
    metadata_path = load_path.replace('.model', '_metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = {}
    
    # Create tokenizer
    tokenizer = SentencePieceTokenizer(
        model_path=load_path,
        vocab_size=metadata.get('vocab_size', 8000),
        character_coverage=metadata.get('character_coverage', 0.9995)
    )
    
    return tokenizer
