"""52%-bar math recalibrated on the v37 iNat-proto REAL submissions.

Two real anchor points now exist on the iNat-proto weight curve (seen flat at 78.95,
gate unchanged, so u_eject=0.535 held from v36):

  config   proxy(S0+w*img)  real_unseen  real_overall  b_cond=unseen/0.535
  v36       31.45            10.42        49.04         0.1948
  w1.5      41.11            12.89        50.12         0.2409
  w2        42.23            13.35        50.32         0.2495

This supersedes the pre-submission anchor 0.584 (HANDOFF §7) with the MEASURED
proxy->real transfer. Writes outputs/target52_math_v37.json.
"""
import json
import os

W_S, W_U = 0.5635, 0.4365
TARGET = 0.52
U_EJ = 0.535           # unseen-route recall @ f=0.72 (gate unchanged since v36)
S_KEEP = 0.918         # seen kept-fraction @ f=0.72
SEEN = 0.7895          # flat across v36/w1.5/w2
A_KEPT = SEEN / S_KEEP # 0.860

# ---- measured points: (tag, proxy_blend, real_unseen, real_overall) ----
pts = [
    ('v36',  31.45, 0.1042, 0.4904),
    ('w1.5', 41.11, 0.1289, 0.5012),
    ('w2',   42.23, 0.1335, 0.5032),
]
proxy_more = {'w3': 43.53}  # measured proxy (research/inat_proto_proxy.py), unbuilt

res = {'weights': {'w_seen': W_S, 'w_unseen': W_U}, 'u_eject': U_EJ, 'A_kept': round(A_KEPT, 4),
       'points': [{'tag': t, 'proxy': p, 'unseen': u, 'overall': o,
                   'b_cond': round(u / U_EJ, 4)} for t, p, u, o in pts]}

# ---- proxy->real transfer (unseen-pop points per proxy point) ----
def slope(a, b):
    return (b[2] - a[2]) / (b[1] - a[1])
res['transfer'] = {
    'v36->w1.5_unseen_per_proxy': round(slope(pts[0], pts[1]) * 100, 4),
    'w1.5->w2_unseen_per_proxy': round(slope(pts[1], pts[2]) * 100, 4),
    'v36->w2_unseen_per_proxy': round((pts[2][2] - pts[0][2]) / (pts[2][1] - pts[0][1]) * 100, 4),
    'v36->w2_overall_per_proxy': round((pts[2][3] - pts[0][3]) / (pts[2][1] - pts[0][1]) * 100, 4),
    'note': 'anchor ~0.27 unseen-pt / proxy-pt (steeper at higher w); NOT the pre-sub 0.584',
}
# global unseen-per-proxy anchor (v36->w2), used for projections
ANCHOR = (pts[2][2] - pts[0][2]) / (pts[2][1] - pts[0][1])  # 0.00272 /proxy-pt

def project(proxy):
    unseen = pts[0][2] + ANCHOR * (proxy - pts[0][1])
    overall = W_S * SEEN + W_U * unseen
    return round(unseen, 4), round(overall, 4)

# ---- project w2.5 (interp proxy) and w3 ----
u3, o3 = project(proxy_more['w3'])
u25, o25 = project((42.23 + 43.53) / 2)  # ~42.88
res['projections'] = {
    'w2.5_proxy~42.9': {'unseen': u25, 'overall': o25},
    'w3_proxy43.53': {'unseen': u3, 'overall': o3},
    'note': 'weight-only saturates ~50.5-50.7%; proxy covered-subset caps ~52 so aggregate is COVERAGE-limited',
}

# ---- what 52% requires ----
need_unseen = (TARGET - W_S * SEEN) / W_U
res['need_for_52'] = {
    'need_unseen_pop': round(need_unseen, 4),
    'have_unseen_pop(w2)': pts[2][2],
    'unseen_gap_pt': round((need_unseen - pts[2][2]) * 100, 2),
    'need_b_cond': round(need_unseen / U_EJ, 4),
    'have_b_cond(w2)': round(pts[2][2] / U_EJ, 4),
    'b_cond_lift_pt': round((need_unseen / U_EJ - pts[2][2] / U_EJ) * 100, 2),
    'need_proxy_blend': round(pts[0][1] + (need_unseen - pts[0][2]) / ANCHOR, 1),
    'max_measured_proxy(w3)': proxy_more['w3'],
    'covered_subset_proxy_ceiling(w2)': 51.36,   # inat_proto_proxy covered_subset
    'cv_covered_gold(w3)': 52.52,
}

# ---- coverage lever: if covered-subset quality held at higher coverage ----
# aggregate proxy ~= cov*proxy_covered + (1-cov)*proxy_uncovered.
# w2 point: cov=0.518, agg=42.23; covered-subset acc=51.36 -> impute uncovered:
cov0, agg0, sub0 = 0.518, 42.23, 51.36
unc = (agg0 - cov0 * sub0) / (1 - cov0)
res['coverage_lever'] = {'covered_frac_now': cov0, 'covered_subset_acc': sub0,
                         'implied_uncovered_acc': round(unc, 2)}
for cov in [0.65, 0.80, 0.90]:
    agg = cov * sub0 + (1 - cov) * unc
    u, o = project(agg)
    res['coverage_lever'][f'cov={cov}'] = {'agg_proxy': round(agg, 2), 'proj_unseen': u,
                                           'proj_overall': o, 'hits_52': o >= TARGET}

os.makedirs('outputs', exist_ok=True)
json.dump(res, open('outputs/target52_math_v37.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
print('\nwrote outputs/target52_math_v37.json')
