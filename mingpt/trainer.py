"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
import sys
import os
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

# Handle imports for both direct execution and module import
try:
    from mingpt.utils import CfgNode as CN
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'mps'
        # dataloder parameters
        C.num_workers = 1
        # evaluation dataloader workers (set to 0 to avoid spawning processes repeatedly)
        C.eval_num_workers = 0
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 4
        #C.block_size = 128
        C.learning_rate = 3e-1
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        # evaluation/validation
        C.eval_interval = 10
        C.eval_iters = 200
        C.val_split = 0.05
        # periodic memory cleanup (helps on MPS/CPU where allocator grows)
        C.cleanup_interval = 500
        
        return C

    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        use_cuda = torch.cuda.is_available()
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=use_cuda,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # free temporary tensors asap
            del logits
            del x
            del y
            del batch

            # optional periodic memory cleanup
            if getattr(self.config, 'cleanup_interval', 0):
                if (self.iter_num + 1) % self.config.cleanup_interval == 0:
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
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

    def estimate_loss(self, dataset=None, eval_iters=None):
        """
        Estimate average loss over eval_iters batches from the provided dataset
        (or the trainer's validation dataset if none provided).
        """
        ds = dataset if dataset is not None else self.val_dataset
        if ds is None:
            return None

        iters = eval_iters if eval_iters is not None else self.config.eval_iters

        use_cuda = torch.cuda.is_available()
        eval_loader = DataLoader(
            ds,
            sampler=torch.utils.data.RandomSampler(
                ds, replacement=True, num_samples=iters * self.config.batch_size
            ),
            shuffle=False,
            pin_memory=use_cuda,
            batch_size=self.config.batch_size,
            num_workers=getattr(self.config, 'eval_num_workers', 0),
            persistent_workers=False,
        )

        self.model.eval()
        losses = []
        with torch.no_grad():
            # iterate exactly `iters` batches by exhausting the loader
            for k, batch in enumerate(eval_loader):
                batch = [t.to(self.device) for t in batch]
                x, y = batch
                _, loss = self.model(x, y)
                losses.append(loss.item())
                # free per-batch eval tensors to keep RAM flat
                del x
                del y
                del batch
        self.model.train()
        # help the DataLoader release any worker resources promptly
        del eval_loader
        # encourage allocator to release cached blocks (useful on MPS/CPU)
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
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if not losses:
            return None
        return sum(losses) / len(losses)
