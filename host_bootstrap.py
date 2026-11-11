import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
df = C.data_for_protocol('P4')
Y, H, P = [], [], {}
for s in range(5):
    tr, te = C.split_protocol(df, 'P4', s)
    z = np.load(f'./results_pred/P4_bin_{s}_pred.npz')
    y = z['y']; assert len(y) == len(te) and (y == df['label'].values[te]).all()
    Y.append(y); H.append(df['src_ip'].values[te])
    for k in z.files:
        if k != 'y': P.setdefault(k, []).append(z[k])
y = np.concatenate(Y); h = np.concatenate(H)
pred = {k: ((lambda a: a[:, 1] if a.ndim == 2 else a)(np.concatenate(v)) >= 0.5).astype(int) for k, v in P.items()}
print('models', list(pred), 'flows', len(y))
hosts, inv = np.unique(h, return_inverse=True)
print('test hosts', len(hosts), 'benign hosts', len(np.unique(h[y == 0])), 'attack hosts', len(np.unique(h[y == 1])))
# per-host confusion counts
def counts(p):
    c = np.zeros((len(hosts), 4))
    for j, (yt, pt) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        np.add.at(c[:, j], inv, ((y == yt) & (p == pt)).astype(float))
    return c  # tn fp fn tp
def mcc(c):
    tn, fp, fn, tp = c
    d = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn - fp*fn) / d if d > 0 else 0.0
C_ = {k: counts(v) for k, v in pred.items()}
for k in C_: print(k, round(mcc(C_[k].sum(0)), 4))
rng = np.random.default_rng(0); B = 2000
# stratified cluster bootstrap: resample benign-only hosts and attack hosts separately so each replicate has both classes
ben = np.where(C_['XGB_all'][:, 2] + C_['XGB_all'][:, 3] == 0)[0]; att = np.setdiff1d(np.arange(len(hosts)), ben)
print('benign-only hosts', len(ben), 'hosts with attacks', len(att))
W = np.zeros((B, len(hosts)))
for b in range(B):
    np.add.at(W[b], rng.choice(ben, len(ben)), 1); np.add.at(W[b], rng.choice(att, len(att)), 1)
out = {}
for c in ['H_stack_MI', 'H_stack']:
    for o in ['XGB_all', 'LGBM', 'RF', 'XGB_MI', 'H_insample']:
        d = np.array([mcc(W[b] @ C_[c]) - mcc(W[b] @ C_[o]) for b in range(B)])
        pt = mcc(C_[c].sum(0)) - mcc(C_[o].sum(0))
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[f'{c}_vs_{o}'] = dict(diff=pt, ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p=float(min(p, 1.0)))
        print(c, o, round(pt, 3), np.round(np.percentile(d, [2.5, 97.5]), 3), round(p, 3))
json.dump(dict(B=B, n_hosts=int(len(hosts)), n_benign_hosts=int(len(ben)), n_attack_hosts=int(len(att)), comparisons=out),
          open('./host_bootstrap.json', 'w'), indent=1)
