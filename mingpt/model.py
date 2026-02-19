"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from mingpt.utils import CfgNode as CN
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class SWIGLU(nn.Module):
    """
    Implementation of SWIGLU (Swish-Gated Linear Unit) activation function.
    SWIGLU(x) = SiLU(Wx) * Vx
    where SiLU is the Swish activation function: SiLU(x) = x * sigmoid(x)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_gate = nn.Linear(in_features, out_features)
        self.w_value = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # SiLU(Wx)
        value = self.w_value(x)        # Vx
        return gate * value            # SiLU(Wx) * Vx


class DAPE(nn.Module):
    """
    DAPE (Data-dependent Attention Positional Encoding) attention bias module.
    Combines attention logits and a base bias along the head dimension and
    produces a learned additive bias per head for each (query, key) position.

    Expected shapes:
    - attention: (B, H, T, T)
    - bias:      (1, H, T, T) or (B, H, T, T)
    Returns:
    - learned additive bias with shape (B, H, T, T)
    """
    def __init__(self, num_heads: int, mlp_width: int = 8):
        super().__init__()
        in_features = 2 * num_heads
        self.mlp = nn.Sequential(
            SWIGLU(in_features, mlp_width),
            nn.Linear(mlp_width, num_heads),
        )

    def forward(self, attention: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, H, Tq, Tk = attention.shape
        # Tile bias across batch if needed
        if bias.dim() == 4 and bias.size(0) == 1:
            bias_tiled = bias.expand(B, -1, -1, -1)
        else:
            bias_tiled = bias
        # Concatenate along head dimension: (B, 2H, T, T)
        concat = torch.cat((attention, bias_tiled), dim=1)
        # Rearrange to apply MLP on the head-channel dimension: (B, T, T, 2H)
        concat = concat.permute(0, 2, 3, 1).contiguous()
        Bt, T1, T2, C = concat.shape
        concat_flat = concat.view(Bt * T1 * T2, C)
        out_flat = self.mlp(concat_flat)
        out = out_flat.view(Bt, T1, T2, H)
        # Back to (B, H, T, T)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # attention type selection
        self.attn_type = getattr(config, 'attn_type', 'standard')
        assert self.attn_type in ('standard', 'alibi'), "attn_type must be 'standard' or 'alibi'"
        # Build per-head causal mask: alternate heads between full causal and a causal diagonal band
        block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        band_width = getattr(config, 'attn_band_width', 5)

        # Optional DAPE bias module
        self.use_dape = bool(getattr(config, 'use_dape', False))
        if self.use_dape:
            dape_mlp_width = int(getattr(config, 'dape_mlp_width', 32))
            self.dape = DAPE(num_heads=self.n_head, mlp_width=dape_mlp_width)
            # Preallocate a reusable base bias buffer (zeros)
            base_bias = torch.zeros(1, self.n_head, block_size, block_size)
            self.register_buffer("dape_base_bias", base_bias, persistent=False)
            # Mode: 'add' vs 'replace'
            mode = getattr(config, 'dape_mode', 'add')
            assert mode in ('add', 'replace'), "dape_mode must be 'add' or 'replace'"
            self.dape_mode = mode
        else:
            self.dape = None
            self.dape_mode = 'add'

        i = torch.arange(block_size).view(block_size, 1)
        j = torch.arange(block_size).view(1, block_size)
        diff = i - j
        full_causal = (diff >= 0)  # lower-triangular including diagonal
        band_causal = (diff >= 0) & (diff < band_width)
        
        per_head_masks = []
        for head_index in range(self.n_head):
            
            if head_index % 2 == 0:
                per_head_masks.append(full_causal)
            else:
                per_head_masks.append(band_causal)
                
        bias = torch.stack(per_head_masks, dim=0).unsqueeze(0)  # (1, n_head, T, T)
        self.register_buffer("bias", bias)

        # ALiBi slopes buffer (only used when attn_type == 'alibi')
        if self.attn_type == 'alibi':
            slopes = self._get_alibi_slopes(self.n_head)
            self.register_buffer("alibi_slopes", slopes.view(1, self.n_head, 1, 1), persistent=False)
        else:
            # placeholder to keep attribute presence consistent
            self.register_buffer("alibi_slopes", torch.tensor(0.0), persistent=False)

    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
        """
        Return per-head ALiBi slopes following the reference implementation.
        """
        import math as _math
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(_math.log2(n) - 3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        if (_math.log2(n_heads)).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** _math.floor(_math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra = CausalSelfAttention._get_alibi_slopes(2 * closest_power_of_2)
            slopes += extra[0::2][:n_heads - closest_power_of_2].tolist()
        return torch.tensor(slopes, dtype=torch.float32)

    def set_attention_type(self, attn_type: str):
        """Dynamically switch attention type at runtime."""
        assert attn_type in ('standard', 'alibi')
        self.attn_type = attn_type
        if attn_type == 'alibi':
            slopes = self._get_alibi_slopes(self.n_head).to(self.bias.device)
            self.register_buffer("alibi_slopes", slopes.view(1, self.n_head, 1, 1), persistent=False)
        else:
            # reset placeholder
            self.register_buffer("alibi_slopes", torch.tensor(0.0, device=self.bias.device), persistent=False)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.use_dape:
            # Slice the preallocated base bias to current T without unnecessary copies
            base_bias = self.dape_base_bias[:, :, :T, :T]
            if base_bias.device != att.device:
                base_bias = base_bias.to(att.device)
            if base_bias.dtype != att.dtype:
                base_bias = base_bias.to(att.dtype)
            learned_bias = self.dape(att, base_bias)
            if self.dape_mode == 'add':
                att = att + learned_bias
            else:  # 'replace'
                att = learned_bias

        # Optional ALiBi biasing (adds linear distance bias per head)
        if self.attn_type == 'alibi':
            i = torch.arange(T, device=att.device).view(1, 1, T, 1)
            j = torch.arange(T, device=att.device).view(1, 1, 1, T)
            dist = (i - j).to(att.dtype)
            alibi = -self.alibi_slopes.to(att.dtype) * dist
            att = att + alibi

        # Apply causal/banded mask after adding any learned bias (in-place to reduce allocs)
        att.masked_fill_(self.bias[:, :self.n_head, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            swiglu  = SWIGLU(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.swiglu(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # attention mask configuration
        C.attn_band_width = 5
        # DAPE options
        C.use_dape = False
        C.dape_mlp_width = 8
        # dape_mode: 'add' to add learned bias to logits; 'replace' to use learned bias only
        C.dape_mode = 'add'
        # attention variant: 'standard' (vanilla) or 'alibi' (linear distance bias)
        C.attn_type = 'alibi'
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        #torch.set_default_dtype(torch.bfloat16)
        print(torch.get_default_dtype())
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192), # 3.86M params
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=1024),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))
        print("ram usage: %.2fGB" % (torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else sum(p.numel() * p.element_size() for p in self.parameters()) / 1e9))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, fused = torch.bfloat16)
        return optimizer

    def set_attention_type(self, attn_type: str):
        """Propagate attention type change to all blocks at runtime."""
        for block in self.transformer.h:
            block.attn.set_attention_type(attn_type)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
