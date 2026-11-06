# hardware: AMD EPYC 7R13, 2 threads, torch 2.15.0+cu132
# command: python labs/thinking-in-tokens/scripts/02_cost_of_thought.py

## 1. Formula vs torch FLOP counter (TinyLLM 'tiny' preset, fp32, batch 1, cached decode)

matmul weights per token N = 4,292,608 (attention + MLP + LM head)

| prompt P | thinking K | counted FLOPs | 2NK + 4Ld(PK+K(K+1)/2) | post's K^2/2 form | attention share | KV bytes held | KV formula |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 139,771,904 | 139,771,904 | 139,722,752 | 1.7% | 98,304 | 98,304 |
| 64 | 256 | 2,500,591,616 | 2,500,591,616 | 2,499,805,184 | 12.1% | 983,040 | 983,040 |
| 256 | 256 | 2,802,581,504 | 2,802,581,504 | 2,801,795,072 | 21.6% | 1,572,864 | 1,572,864 |
| 16 | 1024 | 12,116,295,680 | 12,116,295,680 | 12,113,149,952 | 27.4% | 3,194,880 | 3,194,880 |

## 2. The post's numbers for a 32B-class dense model (arithmetic from the config, not a measurement)

| quantity | post says | recomputed |
|---|---|---|
| non-embedding params N of the 32B-class config | 32B-class | 31.2 B |
| attention / weight FLOPs per token at 1k context | a few per cent | 2.2% |
| attention / weight FLOPs per token at 32k context | roughly comparable | 69% |
| context where the two terms are equal | - | 47,616 tokens |
| KV cache per token, 64 layers x 8 KV heads x 128, bf16 | 256 KiB | 256 KiB |
| KV cache for 10,000 thinking tokens | about 2.6 GB | 2.62 GB |
| weights in bf16, 32e9 params | 64 GB | 64 GB |
| decode floor per token at 3.3 TB/s | about 19 ms | 19.4 ms |
| decode floor per token at 3.35 TB/s (H100 SXM spec) | - | 19.1 ms |
| 10,000 thinking tokens at that floor | three minutes | 3.2 min |

## 3. Bill for K thinking tokens, 32B-class model, P = 1,000, batch 1 (arithmetic)

| K | weight TFLOPs | attention TFLOPs | attention share | KV cache GB | decode floor (s) |
|---|---|---|---|---|---|
| 1,000 | 62 | 2 | 3% | 0.5 | 19 |
| 4,000 | 250 | 16 | 6% | 1.3 | 76 |
| 10,000 | 624 | 79 | 11% | 2.9 | 191 |
| 32,000 | 1,997 | 713 | 26% | 8.7 | 611 |

The decode floor ignores KV-cache reads, which also grow with K, so real batch-1 latency is higher.

## 4. The token channel between reasoning steps

vocabulary V = 151,669 (len(AutoTokenizer('Qwen/Qwen3-0.6B')))
log2 V = 17.21 bits per step (post: about 17.2)
