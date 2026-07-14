"""Verify torch+CUDA and that a BioCLIP backbone loads."""
import torch

def main():
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
    import open_clip
    # BioCLIP 2 (ViT-L/14). Swap to bioclip-2.5-vith14 for the Huge model.
    name = "hf-hub:imageomics/bioclip-2"
    print("loading", name, "...")
    model, _, preprocess = open_clip.create_model_and_transforms(name)
    tokenizer = open_clip.get_tokenizer(name)
    nparams = sum(p.numel() for p in model.parameters())
    print("loaded OK. params=%.1fM" % (nparams / 1e6))

if __name__ == "__main__":
    main()
