"""
Post-training script: load base weights, reuse SentencePiece tokenizer, train on a new
Hugging Face dataset with fresh optimizer/scheduler, and save a finished model.

Defaults per user request:
- resume_from: /Users/huntergray/Documents/AI/minGPT/postTrain/model.pt
- tokenizer_model: /Users/huntergray/Documents/AI/minGPT/postTrain/tokenizer.model
- save_dir: /Users/huntergray/Documents/AI/minGPT/FinishedModel
- device: mps
- learning_rate: 1e-6, batch_size: 1, max_iters: 1000
- short schedule (warmup + linear decay) tuned to short run
"""

import os
import sys
import math
import json
import time
import argparse
from datetime import datetime

import torch
from torch.utils.data import Dataset

# Support both module and direct execution
try:
    from mingpt.utils import set_seed, setup_logging, CfgNode as CN
    from mingpt.model import GPT
    from mingpt.trainer import Trainer
    from mingpt.sentencepiece_tokenizer import SentencePieceDataset, load_tokenizer
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mingpt.utils import set_seed, setup_logging, CfgNode as CN
    from mingpt.model import GPT
    from mingpt.trainer import Trainer
    from mingpt.sentencepiece_tokenizer import SentencePieceDataset, load_tokenizer

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


class TokenChunkDataset(Dataset):
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
    C.system.seed = 1337
    C.system.work_dir = '/Users/huntergray/Documents/AI/minGPT/FinishedModel'

    # data
    C.data = SentencePieceDataset.get_default_config()
    # Keep user's block size; default matches existing 1024
    C.data.block_size = 1024
    C.data.model_prefix = "sp_model"  # not used when loading existing model
    C.data.tokenized_data_path = None
    # Users can optionally set text_column or combine_columns via CLI

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'  # will be overridden by loaded config if provided

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.device = 'mps'
    C.trainer.learning_rate = 1e-6
    C.trainer.batch_size = 1
    C.trainer.max_iters = 1000
    # short schedule parameters
    C.trainer.warmup_iters = 100
    C.trainer.lr_decay_iters = 900
    C.trainer.min_lr = 1e-6
    # eval
    C.trainer.eval_interval = 50
    C.trainer.eval_iters = 100
    C.trainer.val_split = 0.05

    # resume / tokenizer / dataset references
    C.resume_from = '/Users/huntergray/Documents/AI/minGPT/postTrain/model.pt'
    C.tokenizer_model = '/Users/huntergray/Documents/AI/minGPT/postTrain/tokenizer.model'
    C.hf_dataset = None
    C.hf_split = 'train'
    C.hf_text_column = None
    C.hf_combine_columns = []

    return C


def get_lr_scheduler(optimizer, trainer_cfg):
    def update_lr(iter_num):
        if hasattr(trainer_cfg, 'warmup_iters') and iter_num < trainer_cfg.warmup_iters:
            lr = trainer_cfg.learning_rate * (iter_num / max(1, trainer_cfg.warmup_iters))
        elif hasattr(trainer_cfg, 'lr_decay_iters') and iter_num < (trainer_cfg.warmup_iters + trainer_cfg.lr_decay_iters):
            decay_ratio = (iter_num - trainer_cfg.warmup_iters) / max(1, trainer_cfg.lr_decay_iters)
            lr = trainer_cfg.learning_rate - (trainer_cfg.learning_rate - trainer_cfg.min_lr) * decay_ratio
        else:
            lr = trainer_cfg.min_lr
        for g in optimizer.param_groups:
            g['lr'] = lr
        return lr
    return update_lr


def load_base_config_from_dir(base_dir):
    cfg_path = os.path.join(base_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return json.load(f)
    return None


def main(argv=None):
    torch.set_default_dtype(torch.bfloat16)

    config = get_config()

    parser = argparse.ArgumentParser(description='Post-train GPT with existing weights and tokenizer')
    parser.add_argument('--hf_dataset', type=str, default=None, help='Hugging Face dataset name or path')
    parser.add_argument('--hf_split', type=str, default='train', help='HF split (e.g., train, validation)')
    parser.add_argument('--text_column', type=str, default=None, help='Single text column name')
    parser.add_argument('--combine_columns', type=str, nargs='*', default=None, help='Columns to combine for text')

    parser.add_argument('--resume_from', type=str, default=config.resume_from)
    parser.add_argument('--tokenizer_model', type=str, default=config.tokenizer_model)
    parser.add_argument('--save_dir', type=str, default=config.system.work_dir)

    parser.add_argument('--learning_rate', type=float, default=config.trainer.learning_rate)
    parser.add_argument('--batch_size', type=int, default=config.trainer.batch_size)
    parser.add_argument('--max_iters', type=int, default=config.trainer.max_iters)
    parser.add_argument('--device', type=str, default=config.trainer.device)

    parser.add_argument('--warmup_iters', type=int, default=config.trainer.warmup_iters)
    parser.add_argument('--lr_decay_iters', type=int, default=config.trainer.lr_decay_iters)
    parser.add_argument('--min_lr', type=float, default=config.trainer.min_lr)

    args = parser.parse_args(argv)

    # Merge CLI and base config
    config.trainer.learning_rate = args.learning_rate
    config.trainer.batch_size = args.batch_size
    config.trainer.max_iters = args.max_iters
    config.trainer.device = args.device
    config.trainer.warmup_iters = args.warmup_iters
    config.trainer.lr_decay_iters = args.lr_decay_iters
    config.trainer.min_lr = args.min_lr

    config.resume_from = args.resume_from
    config.tokenizer_model = args.tokenizer_model
    config.system.work_dir = args.save_dir

    config.data.text_column = args.text_column if args.text_column else None
    if args.combine_columns is not None:
        config.data.combine_columns = args.combine_columns
    config.hf_dataset = args.hf_dataset
    config.hf_split = args.hf_split

    setup_logging(config)
    set_seed(config.system.seed)

    # Load tokenizer
    if not os.path.exists(config.tokenizer_model):
        raise FileNotFoundError(f"Tokenizer model not found at {config.tokenizer_model}")
    tokenizer = load_tokenizer(config.tokenizer_model)

    # Load dataset (HF optional at runtime)
    if config.hf_dataset is None:
        raise ValueError("--hf_dataset is required for post-training")
    if load_dataset is None:
        raise RuntimeError("datasets library not available. Install with: pip install datasets")
    print(f"Loading Hugging Face dataset: {config.hf_dataset} (split={config.hf_split})")
    dataset = load_dataset(config.hf_dataset, split=config.hf_split)

    # Build SP dataset with existing tokenizer and desired text extraction
    sp_cfg = SentencePieceDataset.get_default_config()
    sp_cfg.block_size = config.data.block_size
    sp_cfg.vocab_size = tokenizer.get_vocab_size()
    sp_cfg.character_coverage = tokenizer.character_coverage if hasattr(tokenizer, 'character_coverage') else 0.9995
    sp_cfg.model_prefix = config.data.model_prefix
    sp_cfg.tokenized_data_path = None
    if config.data.text_column:
        sp_cfg.text_column = config.data.text_column
        sp_cfg.combine_columns = []
    else:
        sp_cfg.text_column = None
        sp_cfg.combine_columns = list(getattr(config.data, 'combine_columns', []) or [])
    sp_dataset = SentencePieceDataset(sp_cfg, dataset, tokenizer=tokenizer)

    # Train/val split from contiguous tokens
    all_tokens = sp_dataset.encoded_data
    val_ratio = getattr(config.trainer, 'val_split', 0.05)
    split_idx = int(len(all_tokens) * (1.0 - val_ratio))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    train_dataset = TokenChunkDataset(train_tokens, sp_dataset.get_block_size())
    val_dataset = TokenChunkDataset(val_tokens, sp_dataset.get_block_size())

    # Build model config and model
    config.model.vocab_size = sp_dataset.get_vocab_size()
    config.model.block_size = sp_dataset.get_block_size()
    model = GPT(config.model)

    # Load base weights
    if not os.path.exists(config.resume_from):
        raise FileNotFoundError(f"Base weights not found at {config.resume_from}")
    print(f"Loading base weights from {config.resume_from}")
    state_dict = torch.load(config.resume_from, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading state dict: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading state dict: {unexpected}")

    # Trainer
    trainer = Trainer(config.trainer, model, train_dataset, val_dataset=val_dataset)

    # Report tokens per iter and target tokens/param
    total_params = sum(p.numel() for p in model.parameters())
    tokens_per_iter = trainer.config.batch_size * config.model.block_size
    target_tpp = 20
    est_iters_for_target = math.ceil((target_tpp * total_params) / max(1, tokens_per_iter))
    print(f"tokens/iter={tokens_per_iter:,}; params={total_params:,}; iters_for_{target_tpp}tpp≈{est_iters_for_target:,}")

    # Batch end callback: scheduler, logs, periodic save
    def on_batch_end(t):
        if not hasattr(t, 'lr_scheduler'):
            t.lr_scheduler = get_lr_scheduler(t.optimizer, config.trainer)
        current_lr = t.lr_scheduler(t.iter_num)

        if t.iter_num % getattr(config.trainer, 'eval_interval', 50) == 0:
            val_loss = t.estimate_loss()
            progress_percent = (t.iter_num / max(1, t.config.max_iters)) * 100.0
            if val_loss is not None:
                print(f"iter {t.iter_num}: train {t.loss.item():.5f} | val {val_loss:.5f} | lr {current_lr:.2e} | {progress_percent:.1f}%")
            else:
                print(f"iter {t.iter_num}: train {t.loss.item():.5f} | lr {current_lr:.2e} | {progress_percent:.1f}%")

        if t.iter_num % 500 == 0 and t.iter_num > 0:
            save_snapshot(model, config)
            # encourage allocator to release cached blocks at snapshot boundaries
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    trainer.set_callback('on_batch_end', on_batch_end)

    # Run training
    trainer.run()

    # Final save
    save_snapshot(model, config, final=True)


def save_snapshot(model, config, final=False):
    os.makedirs(config.system.work_dir, exist_ok=True)
    tag = 'final' if final else f"iter"
    ckpt_path = os.path.join(config.system.work_dir, 'model.pt')
    torch.save(model.state_dict(), ckpt_path)

    # Save tokenizer reference and config
    cfg_out = {
        'model': config.model.to_dict() if hasattr(config.model, 'to_dict') else dict(vocab_size=config.model.vocab_size, block_size=config.model.block_size),
        'trainer': config.trainer.to_dict() if hasattr(config.trainer, 'to_dict') else {
            'learning_rate': config.trainer.learning_rate,
            'batch_size': config.trainer.batch_size,
            'max_iters': config.trainer.max_iters,
            'device': config.trainer.device,
            'warmup_iters': getattr(config.trainer, 'warmup_iters', None),
            'lr_decay_iters': getattr(config.trainer, 'lr_decay_iters', None),
            'min_lr': getattr(config.trainer, 'min_lr', None),
        },
        'tokenizer_model': config.tokenizer_model,
        'saved_at': datetime.utcnow().isoformat() + 'Z',
    }
    with open(os.path.join(config.system.work_dir, 'config.json'), 'w') as f:
        json.dump(cfg_out, f, indent=2)


if __name__ == '__main__':
    main()


