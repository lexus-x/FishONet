"""Generate synthetic fish images for UNSEEN species using Stable Diffusion.
Uses class descriptions from the competition data (legal — external data permitted).
Strategy: For each unseen species, generate K synthetic images to augment training.
The generated images are used as additional "seen" training data for fine-tuning.

  test:   python src/synth_data.py --n_per_class 2 --max_classes 5 --out outputs/synth_test/
  full:   python src/synth_data.py --n_per_class 20 --out outputs/synth_unseen/
"""
import argparse, os, json, pickle, time
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_per_class', type=int, default=20, help='images per unseen class')
    ap.add_argument('--max_classes', type=int, default=0, help='max classes (0=all unseen)')
    ap.add_argument('--out', default='outputs/synth_unseen', help='output directory')
    ap.add_argument('--model', default='stabilityai/stable-diffusion-2-1', help='SD model')
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('Loading class data...')
    lab = json.load(open('data/dl/label_train.json'))
    seen = set(lab.values())
    
    desc_raw = json.load(open('data/dl/descriptions.json'))
    
    classes = list(pickle.load(open('data/dl/all_classes.pkl', 'rb')))
    print(f'Total classes: {len(classes)}, Seen: {len(seen)}')
    
    unseen = [c for c in classes if c not in seen and c in desc_raw]
    print(f'Unseen with descriptions: {len(unseen)}')
    
    if args.max_classes > 0:
        unseen = unseen[:args.max_classes]
    
    os.makedirs(args.out, exist_ok=True)
    
    print(f'Loading {args.model}...')
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()  # Save VRAM
    
    print(f'Generating {args.n_per_class} images for {len(unseen)} unseen classes...')
    
    total = len(unseen) * args.n_per_class
    generated = 0
    t0 = time.time()
    
    for cls_idx, cls_name in enumerate(unseen):
        cls_dir = os.path.join(args.out, cls_name.replace(' ', '_').replace('/', '_'))
        os.makedirs(cls_dir, exist_ok=True)
        
        # Check if already done
        existing = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
        if len(existing) >= args.n_per_class:
            print(f'[{cls_idx+1}/{len(unseen)}] {cls_name}: already done ({len(existing)} images), skipping')
            generated += args.n_per_class
            continue
        
        desc = desc_raw[cls_name]
        if not desc or len(desc) < 20:
            print(f'[{cls_idx+1}/{len(unseen)}] {cls_name}: insufficient description, skipping')
            continue
        
        # Build prompt: "A professional underwater photograph of [species description]"
        prompt = (
            f"A professional underwater scientific photograph of a {cls_name}. "
            f"{desc[:400]}. "
            f"High resolution, natural lighting, clear water, detailed fins and scales, "
            f"scientific specimen photography style."
        )
        
        generator = torch.Generator('cuda').manual_seed(args.seed + cls_idx)
        
        need = args.n_per_class - len(existing)
        for img_idx in range(need):
            try:
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=generator,
                    height=args.resolution,
                    width=args.resolution,
                )
                image = result.images[0]
                out_path = os.path.join(cls_dir, f'synth_{img_idx:04d}.jpg')
                image.save(out_path, quality=95)
                generated += 1
                generator = torch.Generator('cuda').manual_seed(args.seed + cls_idx + img_idx + 1)
            except Exception as e:
                print(f'  ERROR on {cls_name} img {img_idx}: {e}')
                continue
        
        elapsed = (time.time() - t0) / 60
        rate = generated / max(elapsed, 0.01)
        eta = (total - generated) / max(rate, 0.01)
        print(f'[{cls_idx+1}/{len(unseen)}] {cls_name}: {need} generated '
              f'({generated}/{total} total, {rate:.1f} img/min, ETA {eta:.0f}m)', flush=True)
    
    total_time = (time.time() - t0) / 60
    print(f'\nDONE: {generated} images in {total_time:.1f}m '
          f'({generated/total_time:.1f} img/min)')
    
    # Write manifest
    manifest = {}
    for cls_name in unseen:
        cls_dir = os.path.join(args.out, cls_name.replace(' ', '_').replace('/', '_'))
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
            if files:
                manifest[cls_name] = [os.path.join(cls_dir, f) for f in sorted(files)]
    
    json.dump(manifest, open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
    print(f'Manifest saved: {len(manifest)} classes')

if __name__ == '__main__':
    main()