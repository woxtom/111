import torch

from cs336_basics.model import softmax

# def decoding(model: torch.nn.Module, tokenizer, input: torch.Tensor, max_token_length: int, temperature: float, top_p: float):
#     # no batch
#     output = input
#     eot = tokenizer.encode("<|endoftext|>")[0]
#     while output[-1] != eot and len(output) - len(input) < max_token_length:
#         model.eval()
#         logit = model(output)[-1, :]
#         prob = softmax(logit / temperature, -1)
#         sorted_prob, sorted_indices = torch.sort(prob)
#         cum_prob = torch.cumsum(sorted_prob, dim = -1)
#         sorted_mask = cum_prob - sorted_prob > top_p
#         sorted_prob[sorted_mask] = 0
#         sorted_prob = sorted_prob / sorted_prob.sum()
#         idx = torch.multinomial(sorted_prob, num_samples = 1)
#         output = torch.cat([output, torch.tensor(sorted_indices[idx])], dim = 0)

#     return output
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cuda",
):
    model.eval()
    eot_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop context if it exceeds the model's max sequence length
            logits = model(input_ids)                # (1, T, V)
            next_token_logits = logits[0, -1, :]      # (V,)

            # 1. temperature scaling
            if temperature == 0:
                next_id = torch.argmax(next_token_logits).item()
            else:
                scaled_logits = next_token_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)

                # 2. top-p (nucleus) filtering
                if top_p < 1.0:
                    probs = top_p_filter(probs, top_p)

                next_id = torch.multinomial(probs, num_samples=1).item()

            # 3. stop on <|endoftext|>
            if next_id == eot_id:
                break

            generated_ids.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )

    return tokenizer.decode(generated_ids)


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # keep the smallest prefix whose cumulative prob >= top_p;
    # mask everything strictly beyond that cutoff
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()  # renormalize

    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_idx, sorted_probs)
    return filtered
