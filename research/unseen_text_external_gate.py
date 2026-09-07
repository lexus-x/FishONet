"""Kill / promote gate for external traits (plan step 4).

If traits_provided did not clear kill criteria, do NOT fetch FishBase/WoRMS.
Writes outputs/unseen_text_external_decision.json and updates DISCLOSURE.md note.
"""
from __future__ import annotations

import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
RES = os.path.join(OUT, 'unseen_text_traits_provided_results.json')


def main():
    r = json.load(open(RES))
    promote = bool(r.get('promote'))
    decision = {
        'date': str(date.today()),
        'source_results': RES,
        'promote_external_traits': promote,
        'reason': (
            'blend cleared +0.5pt kill criterion — proceed to FishBase/WoRMS'
            if promote else
            'blend best_delta={:.2f} < 0.5pt clear margin; traits alone {:.2f} vs taxon {:.2f}. '
            'External morph text of the same style has negative expected value — KILL text lever.'
            .format(r['verdict']['best_delta'], r['traits_provided'], r['taxon'])
        ),
        'verdict': r.get('verdict'),
        'fishbase_worms_used': False,
    }
    p = os.path.join(OUT, 'unseen_text_external_decision.json')
    json.dump(decision, open(p, 'w'), indent=1)
    print(json.dumps(decision, indent=1), flush=True)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main()
