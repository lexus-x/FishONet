"""Dual Inverted-Softmax (DBNorm) hubness correction for the unseen text-match route.

predict_v12.py's `db()` is a column z-score proxy, not the real two-sided operator
that actually scored v15 real (unseen 15.49%, up from 12.22 with z-debias). That
run was done ad hoc and never saved to src/ -- this persists it so it's reusable
(e.g. for comparing a third backbone's unseen score matrix on hard-sim).

  S: [n_query, n_class] raw cosine-similarity matrix.
  tc: column (class) temperature -- de-hubs popular classes that attract many queries.
  tr: row (query) temperature -- de-hubs queries that are similar to many classes.
  Confirmed-best on this project's hard-sim split: tc=0.05, tr=0.5.
"""
import torch
import torch.nn.functional as F


def dbnorm(S, tc=0.05, tr=0.5):
    return F.log_softmax(S / tc, dim=0) + F.log_softmax(S / tr, dim=1)


def _demo():
    torch.manual_seed(0)
    n_query, n_class = 40, 25
    true_cls = torch.randint(0, n_class, (n_query,))
    S = torch.randn(n_query, n_class) * 0.05
    S[torch.arange(n_query), true_cls] += 1.0   # inject a real signal
    S[:, 0] += 0.3   # a hub class many queries are spuriously close to

    plain_acc = (S.argmax(1) == true_cls).float().mean().item()
    db_acc = (dbnorm(S).argmax(1) == true_cls).float().mean().item()
    assert db_acc >= plain_acc, f'dbnorm should not hurt when a hub is present: plain={plain_acc} db={db_acc}'

    out = dbnorm(S)
    assert out.shape == S.shape
    print(f'demo OK: plain_argmax_acc={plain_acc:.3f} dbnorm_argmax_acc={db_acc:.3f}')


if __name__ == '__main__':
    _demo()

