import open_clip, torch
print("loading ViT-H at force_image_size=336 ...", flush=True)
try:
    m,_,pp = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14', force_image_size=336)
    m = m.cuda().eval()
    import torch
    x = torch.randn(2,3,336,336).cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        f = m.encode_image(x)
    print("OK_336 img_emb", tuple(f.shape), "dim", f.shape[-1])
    print("preprocess_336:", str(pp)[:200])
except Exception as e:
    import traceback; traceback.print_exc(); print("FAIL_336", str(e)[:200])
