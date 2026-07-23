import open_clip, torch, torch.nn as nn
print("loading bioclip-2.5-vith14 ...", flush=True)
m,_,pp = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14')
v = m.visual
print("VISUAL_CHILDREN:", [n for n,_ in v.named_children()])
lin = [n for n,mm in v.named_modules() if isinstance(mm, nn.Linear)]
print("N_LINEAR_VISUAL:", len(lin))
print("FIRST_LIN:", lin[:6])
print("LAST_LIN:", lin[-6:])
# find a transformer block and print its submodule layout
import re
seen=False
for n,mm in v.named_modules():
    if re.search(r'(resblocks|blocks)\.0$', n):
        print("BLOCK0_NAME:", n, "children:", [x for x,_ in mm.named_children()])
        for x,sub in mm.named_children():
            kids=[y for y,_ in sub.named_children()]
            print("   ", x, type(sub).__name__, kids)
        seen=True; break
if not seen: print("no block matched; dumping unique module-type names")
print("HAS_encode_text:", hasattr(m,'encode_text'))
# attn structure (open_clip uses nn.MultiheadAttention with in_proj_weight, not Linear)
for n,mm in v.named_modules():
    if isinstance(mm, nn.MultiheadAttention):
        print("MHA_FOUND:", n, "embed_dim", mm.embed_dim, "has_in_proj_weight", hasattr(mm,'in_proj_weight'))
        break
try:
    import peft; print("PEFT", peft.__version__)
except Exception as e: print("PEFT_NONE", str(e)[:80])
print("PREPROCESS:", str(pp)[:300])
