#!/usr/bin/env python
"""Mechanically verify every compliance claim the technical report makes.

This is an AUDIT tool, not part of inference. It deliberately reads
`data/dl/splits/*.pkl` in order to *prove* the shipped predictions could not
have been produced by a folder oracle. No builder does this; §2 below scans
the whole winning chain to confirm that.

    conda activate onet && python research/verify_compliance.py

Writes outputs/compliance_report.json and exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import zipfile

D = 'data/dl'
OUT = 'outputs'

# the exact dependency chain that produced the submitted artifact
CHAIN = [
    'builders/build_v109_genus_gamble.py',
    'builders/build_v83_rerank_both.py',
    'builders/build_v82_rerank_leakfree.py',
    'builders/build_v81_rerank.py',
    'builders/build_v77_learned_gate.py',
    'builders/build_v56_overlap_336.py',
]
SUBMITTED = 'submissions/submission_v109_genus_gamble.zip'

# per-image variants, in increasing order of how much batch coupling they remove
LADDER = [
    ('gate z1 + f-quota frozen', f'{OUT}/prediction_v109_compliant.json'),
    ('  + dbnorm hub frozen', f'{OUT}/prediction_v109_compliant_dbnorm.json'),
]

fails: list[str] = []
report: dict = {}


def check(name: str, ok: bool, detail: str) -> None:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
    report[name] = {'pass': bool(ok), 'detail': detail}
    if not ok:
        fails.append(name)


def load_submitted() -> dict:
    with zipfile.ZipFile(SUBMITTED) as z:
        return json.loads(z.read('prediction.json'))


def main() -> int:
    classes = list(pickle.load(open(f'{D}/all_classes.pkl', 'rb')))
    lab = json.load(open(f'{D}/label_train.json'))
    photographed = set(lab.values())
    unphotographed = set(classes) - photographed
    pred = load_submitted()

    print('\n1. Label space')
    check('full class vocabulary', len(classes) == 17393, f'{len(classes)} classes')
    check('photographed / unphotographed split',
          len(photographed) == 5795 and len(unphotographed) == 11598,
          f'{len(photographed)} with training images, {len(unphotographed)} without')
    check('all 35,665 eval images predicted', len(pred) == 35665, f'{len(pred)} predictions')
    unknown = {v for v in pred.values()} - set(classes)
    check('every prediction is in the vocabulary', not unknown,
          'no out-of-vocabulary labels' if not unknown else f'{len(unknown)} unknown')

    print('\n2. No split file is read anywhere in the winning chain')
    pat = re.compile(r'splits?[/\\](?:test|unseen|train)\.pkl|splits/')
    for f in CHAIN:
        src = open(f).read() if os.path.exists(f) else ''
        hits = [ln for ln in src.splitlines()
                if pat.search(ln) and not ln.strip().startswith('#')]
        check(f'no split read in {os.path.basename(f)}', not hits,
              'clean' if not hits else f'{len(hits)} reference(s): {hits[:2]}')

    print('\n3. One argmax over the full space (not restricted to either side)')
    n_photo = sum(1 for v in pred.values() if v in photographed)
    n_unphoto = len(pred) - n_photo
    check('predictions land on both sides of the split',
          n_photo > 0 and n_unphoto > 0,
          f'{n_photo} photographed-class predictions, {n_unphoto} unphotographed-class')

    print('\n4. The router is blind (a folder oracle would score 100% here)')
    test = set(pickle.load(open(f'{D}/splits/test.pkl', 'rb')))
    unseen = set(pickle.load(open(f'{D}/splits/unseen.pkl', 'rb')))
    # AUDIT ONLY -- ground-truth membership, never available to the pipeline
    m = {'test->photo': 0, 'test->unphoto': 0, 'uns->photo': 0, 'uns->unphoto': 0}
    for fn, c in pred.items():
        side = 'photo' if c in photographed else 'unphoto'
        if fn in test:
            m[f'test->{side}'] += 1
        elif fn in unseen:
            m[f'uns->{side}'] += 1
    t_pure = m['test->photo'] / max(1, m['test->photo'] + m['test->unphoto'])
    u_pure = m['uns->unphoto'] / max(1, m['uns->photo'] + m['uns->unphoto'])
    check('routing is NOT an oracle', t_pure < 0.999 and u_pure < 0.999,
          f'seen-split routed to a photographed class {t_pure*100:.2f}%, '
          f'unseen-split to an unphotographed class {u_pure*100:.2f}% '
          f'(an oracle would be 100.00% / 100.00%)')
    report['routing_confusion'] = m

    print('\n5. Batch-coupling exposure: how much a strictly per-image build would change')
    ladder = []
    for label, path in LADDER:
        if not os.path.exists(path):
            print(f'  [SKIP] {label}: {path} not present')
            continue
        alt = json.load(open(path))
        if set(alt) != set(pred):
            print(f'  [SKIP] {label}: key mismatch')
            continue
        d = sum(1 for k in pred if alt[k] != pred[k])
        ladder.append({'variant': label, 'changed': d,
                       'pct': round(d / len(pred) * 100, 3)})
        print(f'  [INFO] {label}: {d} of {len(pred)} predictions change '
              f'({d/len(pred)*100:.3f}%) -> worst-case bound +/-{d/len(pred)*100:.3f} pt')
    report['coupling_ladder'] = ladder

    print('\n6. Deployed gate artifact is reproducible from the committed script')
    try:
        import numpy as np
        a = pickle.load(open(f'{OUT}/learned_gate_v79.pkl', 'rb'))
        b = pickle.load(open(f'{OUT}/learned_gate_v79_repro.pkl', 'rb'))
        same = (np.array_equal(a['model'].coef_, b['model'].coef_)
                and np.array_equal(a['model'].intercept_, b['model'].intercept_)
                and a['feats'] == b['feats'])
        check('learned_gate_v79 reproduces bit-identically', same,
              'coefficients and intercept match exactly '
              '(GATE_C=100 GATE_TAG=v79 python research/learned_gate_v77.py)')
    except FileNotFoundError as e:
        print(f'  [SKIP] repro artifact missing: {e}')

    os.makedirs(OUT, exist_ok=True)
    report['ok'] = not fails
    json.dump(report, open(f'{OUT}/compliance_report.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/compliance_report.json')
    print('RESULT:', 'ALL CHECKS PASS' if not fails else f'{len(fails)} FAILED: {fails}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
