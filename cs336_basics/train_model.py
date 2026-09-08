import argparse
import os
import typing

import numpy as np
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
from pathlib import Path
import time
from cs336_basics.model import transformer_lm

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # l = -(o[x_i+1] - log(sum(exp(x_a, a = 1...vocab_size))))
    inputs = inputs - torch.amax(inputs, -1, keepdim = True)
    loss = - (torch.sum(torch.nn.functional.one_hot(targets, num_classes = inputs.shape[-1]) * inputs, -1, keepdim = False) - torch.log(torch.sum(torch.exp(inputs), -1)))
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), weight_decay = 1e-2, eps = 1e-8) -> None:
        # learning rate alpha
        # beta1 beta2
        # weight decay rate lambda
        # stability factor epsilon
        if len(betas) != 2:
            raise ValueError(f"Invalid hyperpamater: {betas}")
        if lr <= 0 or eps <= 0:
            raise ValueError("Invalid hyperparameter.")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros(p.data.shape, dtype = p.data.dtype, device = p.data.device))
                v = state.get("v", torch.zeros(p.data.shape, dtype = p.data.dtype, device = p.data.device))
                grad = p.grad.data
                alpha_t = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                # weight decay
                p.data -= lr * weight_decay * p.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad ** 2
                # weight update
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

# adamw_accounting
#
# parameters + gradient + optimizer state (1 + 1 + 2)
#
# embedding
#   vocab_size * d_model
# num_layers * transformer block
#   2 * rmsnorm block
#       d_model
#   msa with rope
#       d_model * d_model * 4
#   ffn
#       d_model * d_model * 3 * 8/3
# final RMSNorm
#   d_model
# output_embedding
#   vocab_size * d_model
#
# activations with BATCH
#
# num_layers * transformer_block
#   rmsnorm * 2
#       context_length * d_model
#   msa
#       qkv
#           context_length * d_model
#       qk^T matrix multiply
#           context_length * d_model * 2
#       softmax
#           context_length * context_length * num_heads
#       weighted sum of values
#           context_length * d_model
#       output projection
#           context_length * d_model
#   swiglu
#       context_length * d_model <- x
#       context_length * d_model * 8/3 * 4
# final rmsnorm
#   context_length * d_model
# output_embedding
#   context_length * d_model
# cross-entropy on logits
#   context_length * vocab_size

def learning_rate_schedule(it: int, max_learning_rate: float, min_learing_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif it < cosine_cycle_iters:
        return min_learing_rate + (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) / 2 * (max_learning_rate - min_learing_rate)
    else:
        return min_learing_rate

def gradient_clipping(parameters, max_l2_norm):
    total_sq_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq_norm += p.grad.data.norm(2).item()**2
    total_norm = total_sq_norm ** 0.5
    if total_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= max_l2_norm / ( 1e-6 + total_norm)

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    n = len(x)
    max_start = n - context_length  # exclusive upper bound
    if max_start < 1:
        raise ValueError(f"len(x)={n} must be > context_length={context_length} to sample at least one (input, target) pair")

    starts = np.random.randint(0, max_start, size=batch_size)  # sampled w/ replacement

    inputs = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + 1 + context_length] for s in starts])

    inputs = torch.tensor(inputs).long().to(device)
    targets = torch.tensor(targets).long().to(device)
    return inputs, targets

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM")

    g = p.add_argument_group("model")
    g.add_argument("--vocab-size", type=int, required=True)
    g.add_argument("--context-length", type=int, required = True)
    g.add_argument("--num-layers", type=int, required = True)
    g.add_argument("--d-model", type=int, required = True)
    g.add_argument("--num-heads", type=int, required = True)
    g.add_argument("--d-ff", type=int)
    g.add_argument("--theta", type=float)

    g = p.add_argument_group("optimizer")
    g.add_argument("--max-lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=3e-5)
    g.add_argument("--warmup-steps", type=int, default=200)
    g.add_argument("--cosine-cycle-iters", type = int, default = 1000)
    g.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--eps", type=float, default=1e-8)
    g.add_argument("--grad-clip", type=float, default=1.0)

    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, required=True)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dtype", type=str, default="float32", choices=DTYPES.keys())
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("data / io")
    g.add_argument("--train-dataset-path", type=Path, required=True)
    g.add_argument("--validation-dataset-path", type=Path, required=True)
    g.add_argument("--data-dtype", type=str, default="uint16")
    g.add_argument("--checkpoint-path", type=Path, required=True, help="directory to write checkpoints into")
    g.add_argument("--resume-from", type=Path, default=None)

    g = p.add_argument_group("logging / eval")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--eval-every", type=int, default=200)
    g.add_argument("--eval-batches", type=int, default=20)
    g.add_argument("--checkpoint-every", type=int, default=1000)
    g.add_argument("--wandb-project", type=str, default=None)

    args = p.parse_args()
    return args


@torch.no_grad()
def evaluate(model, data, args):
    """Average loss over a handful of random validation batches."""
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.context_length, args.device)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)

    # ---- data (memmap: nothing is read into RAM until you index it) ----
    train_data = np.memmap(args.train_dataset_path, dtype=np.dtype(args.data_dtype), mode="r")
    val_data = np.memmap(args.validation_dataset_path, dtype=np.dtype(args.data_dtype), mode="r")
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    # ---- model / optimizer ----
    model = transformer_lm(
        args.vocab_size, args.context_length, args.num_layers, args.d_model,
        args.num_heads, args.d_ff, args.theta, args.device, DTYPES[args.dtype],
    )
    opt = AdamW(model.parameters(), lr=args.max_lr, betas=tuple(args.betas),
                weight_decay=args.weight_decay, eps=args.eps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M")

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, opt)
        print(f"resumed from {args.resume_from} at step {start_step}")

    # ---- optional wandb ----
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    def log(metrics, step):
        print(f"step {step:6d} | " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                for k, v in metrics.items()))
        if wandb_run:
            wandb_run.log(metrics, step=step)

    # ---- training loop ----
    model.train()
    t0 = time.time()
    for step in range(start_step, args.steps):
        # 1. learning rate for this step
        lr = learning_rate_schedule(step, args.max_lr, args.min_lr, args.warmup_steps, args.cosine_cycle_iters)
        for group in opt.param_groups:
            group["lr"] = lr

        # 2. forward / backward / update
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        loss = cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        opt.step()

        # 3. logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            log({"train/loss": loss.item(), "lr": lr, "elapsed_s": elapsed}, step)

        if step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_data, args)
            log({"val/loss": val_loss, "val/perplexity": math.exp(val_loss)}, step)

        # 4. checkpointing
        if step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, opt, step, args.checkpoint_path / f"step_{step}.pt")

    # final eval + checkpoint
    val_loss = evaluate(model, val_data, args)
    log({"val/loss": val_loss, "val/perplexity": math.exp(val_loss)}, args.steps)
    save_checkpoint(model, opt, args.steps, args.checkpoint_path / "final.pt")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
