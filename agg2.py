"""Aggregate the second-dataset (Edge-IIoTset) and robustness experiments."""
import json, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = {}
ORDER = ['LR', 'RF', 'LGBM', 'XGB_all', 'XGB_MI', 'MLP', 'CNN', 'H_insample', 'H_stack_MI', 'H_stack']


def summarize(files):
    rows = {}
    for f in files:
        r = json.load(open(f))
        if not r.get('done'): continue
        for m, v in r['models'].items():
            rows.setdefault(m, []).append(v)
    return {m: dict(n=len(v), mcc=float(np.mean([x['mcc'] for x in v])), mcc_sd=float(np.std([x['mcc'] for x in v], ddof=1)) if len(v) > 1 else 0.0,
                    f1m=float(np.mean([x['f1_macro'] for x in v])), f1m_sd=float(np.std([x['f1_macro'] for x in v], ddof=1)) if len(v) > 1 else 0.0,
                    acc=float(np.mean([x['acc'] for x in v])), per_class=np.mean([x['per_class_f1'] for x in v], 0).tolist() if 'per_class_f1' in v[0] else None)
            for m, v in rows.items()}


E = 'edge'
OUT['edge_P1_multi'] = summarize(sorted(glob.glob(f'{E}/res_std/P1_multi_*.json')))
OUT['edge_P3_multi'] = summarize(sorted(glob.glob(f'{E}/res_std/P3_multi_*.json')))
OUT['edge_P3s_multi'] = summarize(sorted(glob.glob(f'{E}/res_strict/P3_multi_*.json')))
OUT['edge_P3s_bin'] = summarize(sorted(glob.glob(f'{E}/res_strict/P3_bin_*.json')))
for k in ['edge_P1_multi', 'edge_P3s_multi']:
    fs = sorted(glob.glob(f'{E}/res_{"std" if "P1" in k else "strict"}/P{"1" if "P1" in k else "3"}_multi_*.json'))
    if fs:
        r = json.load(open(fs[0])); OUT[k + '_classes'] = r.get('classes'); OUT[k + '_n'] = r['n']; OUT[k + '_k'] = r['mi']['k_attributes']
        OUT[k + '_mi_top'] = list(r['mi']['ranking'])[:10]
# paired stats on edge multi (H_stack vs others, per seed)
for k, d in [('edge_P1_multi', f'{E}/res_std/P1_multi_*.json'), ('edge_P3s_multi', f'{E}/res_strict/P3_multi_*.json')]:
    runs = [json.load(open(f)) for f in sorted(glob.glob(d))]
    if runs:
        OUT[k + '_wins'] = {m: int(sum(r['models']['H_stack']['mcc'] > r['models'][m]['mcc'] for r in runs if m in r['models'])) for m in ORDER if m != 'H_stack'}
# Edge unseen family
zd = {}
for f in sorted(glob.glob(f'{E}/res_strict/ZD_*_0.json')):
    r = json.load(open(f)); zd[r['family']] = dict(n=r['n_fam'], **r['models'])
OUT['edge_zeroday'] = zd

# ---- ToN-IoT P4 paired bootstrap on pooled predictions
P = './results_pred'
preds = [np.load(f) for f in sorted(glob.glob(f'{P}/P4_bin_*_pred.npz'))]
if len(preds) == 5:
    LAB = [{m: p[m].argmax(1) for m in p.files if m != 'y'} for p in preds]
    def mcc_pool(idx_list, m):
        tp = fp = fn = tn = 0
        for p, L, i in zip(preds, LAB, idx_list):
            y = p['y'][i]; yh = L[m][i]
            tp += int(((y == 1) & (yh == 1)).sum()); fp += int(((y == 0) & (yh == 1)).sum())
            fn += int(((y == 1) & (yh == 0)).sum()); tn += int(((y == 0) & (yh == 0)).sum())
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / den if den else 0.0
    rng = np.random.RandomState(0); B = 1000
    base = [np.arange(len(p['y'])) for p in preds]
    models = [m for m in ['H_stack_MI', 'H_stack', 'XGB_all', 'LGBM', 'RF', 'XGB_MI', 'H_insample'] if m in preds[0].files]
    point = {m: float(mcc_pool(base, m)) for m in models}
    boots = {m: [] for m in models}
    for b in range(B):
        idx = [rng.randint(0, len(p['y']), len(p['y'])) for p in preds]
        for m in models: boots[m].append(mcc_pool(idx, m))
    boots = {m: np.array(v) for m, v in boots.items()}
    comp = {}
    for a in ['H_stack_MI', 'H_stack']:
        for b_ in ['XGB_all', 'LGBM', 'RF', 'XGB_MI', 'H_insample']:
            d = boots[a] - boots[b_]
            comp[f'{a}_vs_{b_}'] = dict(diff=point[a] - point[b_], ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                        p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    OUT['p4_bootstrap'] = dict(point=point, comparisons=comp, B=B)

# ---- dns_query ablation (P4, trees)
nd = {}
for f in sorted(glob.glob('./results_nodns/P4_bin_*.json')):
    r = json.load(open(f))
    for m, v in r['models'].items(): nd.setdefault(m, []).append(v['mcc'])
base4 = {}
for f in sorted(glob.glob('./results/P4_bin_*.json')):
    r = json.load(open(f))
    for m, v in r['models'].items(): base4.setdefault(m, []).append(v['mcc'])
OUT['nodns_P4'] = {m: dict(with_dns=float(np.mean(base4[m])), without=float(np.mean(v)), n=len(v)) for m, v in nd.items() if len(v) == 5}
zdn = {}
for f in sorted(glob.glob('./results_nodns/ZD_*.json')):
    r = json.load(open(f)); zdn[r['family']] = {m: v['detection_rate'] for m, v in r['models'].items()}
OUT['nodns_zeroday'] = zdn

# ---- zero-day seeds
zs = {}
for f in sorted(glob.glob('./results/ZD_*_0.json')) + sorted(glob.glob('./results_zdseeds/ZD_*.json')):
    r = json.load(open(f))
    for m, v in r['models'].items():
        if m == 'H_stack': continue
        zs.setdefault(r['family'], {}).setdefault(m, []).append(v['detection_rate'])
OUT['zd_seeds'] = {fam: {m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v)) for m, v in d.items()} for fam, d in zs.items()}

# ---- CNN input-order permutation (P4)
pm = [json.load(open(f))['models']['CNN']['mcc'] for f in sorted(glob.glob('./results_perm/P4_bin_*.json'))]
OUT['perm_P4_cnn'] = dict(permuted=float(np.mean(pm)) if pm else None, n=len(pm), original=float(np.mean(base4['CNN'])),
                          per_fold_perm=pm, per_fold_orig=base4['CNN'])
json.dump(OUT, open('./agg2.json', 'w'), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) or len(json.dumps(v)) < 400 else '...') for k, v in OUT.items()}, indent=0)[:3000])
