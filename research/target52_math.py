"""Exact 52%-bar math with v36 real calibration + real gate ROC.

Deliverable #1 of the 52% push. Answers: is 52% under the ACHIEVABLE ceiling
(real gate AUC ~0.91 + current b), under the perfect-gate oracle, and what is the
MINIMUM b_cond / u_recall / gate-AUC needed. Writes outputs/target52_math.json.

All ROC points are the REAL eval-folder ROC of the combined gate (diagnostic only,
never used for routing). Conditionals A/b are the v36 calibration that exactly
reproduces the real submission (seen 78.95 / unseen 10.42 @ f=0.72).
"""
import json
import os

W_S, W_U = 0.5635, 0.4365
TARGET = 0.52

# --- v36 real result (measured on Codabench) ---
v36_seen, v36_unseen = 0.7895, 0.1042
v36_overall = W_S * v36_seen + W_U * v36_unseen  # identity check -> 0.4904

# --- calibration A: kept/ejected-conditional (matches v36 exactly at f=0.72) ---
# tf->seen-cls 0.918 (s_keep), uf->uns-cls 0.535 (u_ej) from routing diagnostic.
s_keep_72, u_ej_72 = 0.918, 0.535
A_kept = v36_seen / s_keep_72     # closed-set acc on KEPT seen
b_cond = v36_unseen / u_ej_72     # unseen-route acc on EJECTED unseen

# --- calibration B: population-conditional constants (path53 ROC framework) ---
A_full = 0.812   # real closed-set acc over ALL seen (kept+ejected)
b_full = 0.178   # real unseen-route acc over ALL unseen

res = {
    'weights': {'w_seen': W_S, 'w_unseen': W_U},
    'v36_real': {'seen': v36_seen, 'unseen': v36_unseen,
                 'overall_identity': round(v36_overall, 4)},
    'calibration_kept_conditional': {'A_kept': round(A_kept, 4), 'b_cond': round(b_cond, 4),
                                     's_keep@f72': s_keep_72, 'u_eject@f72': u_ej_72},
    'calibration_population': {'A_full': A_full, 'b_full': b_full},
}

# ---------- 1. seen fixed at 78.95: needed unseen_pop for 52% ----------
need_unseen_pop = (TARGET - W_S * v36_seen) / W_U
res['need_unseen_pop_at_fixed_seen'] = round(need_unseen_pop, 4)
res['unseen_pop_gap_pt'] = round((need_unseen_pop - v36_unseen) * 100, 2)

# ---------- 2. perfect-gate oracle ceiling (eject ALL unseen, keep ALL seen) ----------
oracle = W_S * A_full + W_U * b_full
res['oracle_ceiling'] = round(oracle, 4)
res['is_52_under_oracle'] = oracle >= TARGET
# under a PERFECT gate, what b_full is needed for 52%?
res['b_full_needed_perfect_gate'] = round((TARGET - W_S * A_full) / W_U, 4)

# ---------- 3. minimum b_cond needed at CURRENT achievable gate ----------
# Real eval-folder ROC of best achievable gate (ctft+fullm) from gate_v36_margin_results.
# (s_kept, u_rec) points measured on eval folders:
roc_best_gate = [
    (0.24, 0.9421, 0.4751), (0.26, 0.9351, 0.5119), (0.28, 0.9275, 0.5478),
    (0.30, 0.9196, 0.5835), (0.32, 0.9102, 0.6172),
]
# path53 combined_fullm extends to heavier eject:
roc_ext = [(0.35, 0.8895, 0.6592), (0.4365, 0.8402, 0.7937), (0.5, 0.7918, 0.8767)]

# For each operating point, solve for b_cond that yields overall=0.52 (A_kept fixed).
min_b_needed = []
for ej, s_kept, u_rec in roc_best_gate + roc_ext:
    # overall = W_S*A_kept*s_kept + W_U*b*u_rec = TARGET
    seen_term = W_S * A_kept * s_kept
    b_needed = (TARGET - seen_term) / (W_U * u_rec)
    proj_now = seen_term + W_U * b_cond * u_rec
    min_b_needed.append({'eject': ej, 's_kept': s_kept, 'u_rec': u_rec,
                         'b_cond_needed_for_52': round(b_needed, 4),
                         'proj_overall_now': round(proj_now, 4)})
res['min_b_cond_needed_per_op'] = min_b_needed
res['best_proj_overall_now'] = round(max(x['proj_overall_now'] for x in min_b_needed), 4)
res['min_b_cond_needed_for_52'] = round(min(x['b_cond_needed_for_52'] for x in min_b_needed), 4)

# ---------- 4. minimum u_recall needed at CURRENT b_cond (need better gate) ----------
# hold s_kept at the value that keeps seen at 78.95 (=0.918); solve u_rec.
u_rec_needed = (TARGET - W_S * A_kept * s_keep_72) / (W_U * b_cond)
res['u_recall_needed_at_current_b_and_seen'] = round(u_rec_needed, 4)
res['u_recall_now@f72'] = u_ej_72

# ---------- 5. proxy translation of the b requirement ----------
# proxy->real anchor: proxy 33.35 (v36 stack+TB) <-> real b_cond 0.1948.
proxy_anchor, real_anchor = 33.35, b_cond
oracle_family_proxy = 35.46  # saturation_check (encoder ceiling w/ PERFECT family)
res['proxy_translation'] = {
    'anchor_proxy': proxy_anchor, 'anchor_real_b': round(real_anchor, 4),
    'proxy_needed_for_min_b': round(proxy_anchor * res['min_b_cond_needed_for_52'] / real_anchor, 2),
    'encoder_oracle_family_proxy_ceiling': oracle_family_proxy,
    'encoder_ceiling_reachable': (proxy_anchor * res['min_b_cond_needed_for_52'] / real_anchor)
                                 <= oracle_family_proxy,
}

# ---------- verdict ----------
res['verdict'] = {
    'is_52_under_achievable_gate_current_b': res['best_proj_overall_now'] >= TARGET,
    'is_52_under_oracle': res['is_52_under_oracle'],
    'min_b_cond_needed_for_52': res['min_b_cond_needed_for_52'],
    'current_b_cond': round(b_cond, 4),
    'b_cond_lift_needed_pt': round((res['min_b_cond_needed_for_52'] - b_cond) * 100, 2),
    'b_cond_lift_needed_rel': round(res['min_b_cond_needed_for_52'] / b_cond - 1, 3),
}

os.makedirs('outputs', exist_ok=True)
json.dump(res, open('outputs/target52_math.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
print('\nwrote outputs/target52_math.json')
