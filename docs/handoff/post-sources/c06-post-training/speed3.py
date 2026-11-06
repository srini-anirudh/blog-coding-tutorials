import time, torch, sys
torch.set_num_threads(int(sys.argv[2]))
from transformers import AutoModelForCausalLM
dev=sys.argv[1]
m = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M', dtype=torch.float32).to(dev)
x = torch.randint(0, 49152, (8, 128), device=dev)
for i in range(3):
    t=time.time(); out = m(input_ids=x, labels=x, use_cache=False); out.loss.backward(); m.zero_grad()
    if dev=='mps': torch.mps.synchronize()
    print(dev, 'fb', round(time.time()-t,2), flush=True)
m.eval()
p=torch.randint(0,49152,(32,64),device=dev)
for i in range(2):
    t=time.time()
    with torch.no_grad(): m.generate(input_ids=p, attention_mask=torch.ones_like(p), max_new_tokens=32, do_sample=True, top_k=0, pad_token_id=0, eos_token_id=None)
    if dev=='mps': torch.mps.synchronize()
    print(dev,'gen 32x32',round(time.time()-t,2),flush=True)
