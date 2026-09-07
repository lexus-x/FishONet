"""Loadability probe for organizer-listed general-biology HF encoders."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402

CANDIDATES = [
    'hf-hub:imageomics/bioclip-2.5-vitb16',
    'hf-hub:imageomics/bioclip-2.5-vitl14',
    'hf-hub:imageomics/bioclip-2.5-vith14',
    'hf-hub:imageomics/bioclip-2',
    'hf-hub:imageomics/TreeOfLife-CLIP',
    'hf-hub:imageomics/Arboretum-CLIP',
    'hf-hub:BGLab/BioTrove-CLIP',
]


def main():
    import open_clip

    rows = []
    for model_id in CANDIDATES:
        row = {'model': model_id}
        t0 = time.time()
        try:
            open_clip.create_model_and_transforms(model_id)
            row['load'] = 'ok'
            row['elapsed_s'] = round(time.time() - t0, 1)
        except Exception as e:
            row['load'] = 'fail'
            row['error'] = str(e)[:240]
        rows.append(row)
        print(row, flush=True)
    out = os.path.join(OUT, 'organizer_encoder_probe.json')
    json.dump(rows, open(out, 'w'), indent=2)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
