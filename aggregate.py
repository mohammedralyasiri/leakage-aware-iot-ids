"""Aggregate results into tables (JSON/CSV) and publication figures."""
import os, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare

R = os.environ.get("RESULTS_DIR", "results")
FIG = os.environ.get("FIG_DIR", "fig"); os.makedirs(FIG, exist_ok=True)
OUT = os.environ.get("AGG_OUT", "agg.json")
ORDER = ['LR', 'RF', 'LGBM', 'XGB_all', 'XGB_MI', 'MLP', 'CNN', 'H_insample', 'H_stack_MI', 'H_stack']
NAMES = {'LR': 'LR', 'RF': 'RF', 'LGBM': 'LightGBM', 'XGB_all': 'XGBoost (all)', 'XGB_MI': 'XGBoost (MI)', 'MLP': 'MLP',
         'CNN': '1D-CNN (MI)', 'H_insample': 'CNN→XGB in-sample', 'H_stack_MI': 'CF-MI-CNN-XGB [X_MI‖P̂]',
         'H_stack': 'CF-MI-CNN-XGB [X‖P̂]'}
CLASSES = ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

runs = []
for f in sorted(glob.glob(f"{R}/P*_*.json")):
    r = json.load(open(f))
    if not r.get('done'): continue
    for m, v in r['models'].items():
        if m.endswith('_diverged'): continue
        runs.append(dict(protocol=r['protocol'], task=r['task'], seed=r['seed'], model=m, **{k: v[k] for k in v if not isinstance(v[k], list)},
                         per_class_f1=v.get('per_class_f1')))
D = pd.DataFrame(runs)
agg = {}


def ms(s):
    return f"{s.mean():.4f} ± {s.std(ddof=1) if len(s) > 1 else 0:.4f}"


# ---- binary summary per protocol
B = D[D.task == 'bin']
tab = {}
for p in sorted(B.protocol.unique()):
    sub = B[B.protocol == p]; tab[p] = {}
    for m in ORDER:
        s = sub[sub.model == m]
        if len(s) == 0: continue
        tab[p][m] = {k: dict(mean=float(s[k].mean()), sd=float(s[k].std(ddof=1)) if len(s) > 1 else 0.0, n=len(s))
                     for k in ['acc', 'mcc', 'f1_macro', 'precision', 'recall', 'fpr', 'auc', 'pr_auc', 'train_s', 'infer_us_per_flow', 'size_kb']}
agg['binary'] = tab

# pooled confusion for P4 (all folds)
pool = {}
for m in ORDER:
    s = B[(B.protocol == 'P4') & (B.model == m)]
    if len(s):
        tn, fp, fn, tp = [float(x) for x in (s.tn.sum(), s.fp.sum(), s.fn.sum(), s.tp.sum())]
        sdiv = lambda a, b: float(a / b) if b else float('nan')
        mcc = sdiv(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        pool[m] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), mcc=mcc, recall=sdiv(tp, tp + fn),
                       fpr=sdiv(fp, fp + tn), precision=sdiv(tp, tp + fp), folds=len(s))
agg['P4_pooled'] = pool

# ---- statistics: paired over all binary runs in P2-P4
S = B[B.protocol.isin(['P2', 'P3', 'P4'])].pivot_table(index=['protocol', 'seed'], columns='model', values='mcc').dropna()
stats = dict(n_pairs=len(S))
ref = 'H_stack'
for m in ORDER:
    if m == ref or m not in S: continue
    d = S[ref] - S[m]
    try:
        p = wilcoxon(S[ref], S[m]).pvalue if (d != 0).any() else 1.0
    except Exception:
        p = float('nan')
    stats[m] = dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=float(p))
ranks = S[[m for m in ORDER if m in S]].rank(axis=1, ascending=False).mean().sort_values()
stats['mean_rank'] = ranks.to_dict()
try:
    stats['friedman_p'] = float(friedmanchisquare(*[S[m] for m in ranks.index]).pvalue)
except Exception:
    pass
agg['stats_mcc'] = stats

# ---- MI / k / cnn gain share / shap
mi = {}
for f in sorted(glob.glob(f"{R}/P*_bin_*.json")):
    r = json.load(open(f))
    if not r.get('done') or 'mi' not in r: continue
    mi[f"{r['protocol']}_{r['seed']}"] = dict(k=r['mi']['k_attributes'], top=list(r['mi']['ranking'].items())[:r['mi']['k_attributes']],
                                              gain_share=r['models'].get('H_stack', {}).get('gain_share_cnn'),
                                              gain_share_mi=r['models'].get('H_stack_MI', {}).get('gain_share_cnn'),
                                              shap=r.get('shap_XGB_MI'), test_counts=r['test_class_counts'], spw=r['scale_pos_weight'])
    if r['protocol'] == 'P4': mi[f"P4_{r['seed']}"]['test_attack_hosts'] = r.get('test_attack_hosts')
agg['runs_meta'] = mi

# ---- multi-class
M = D[D.task == 'multi']
mt = {}
for p in sorted(M.protocol.unique()):
    sub = M[M.protocol == p]; mt[p] = {}
    for m in ORDER:
        s = sub[sub.model == m]
        if len(s) == 0: continue
        pcf = np.array([x for x in s.per_class_f1])
        mt[p][m] = dict(acc=float(s.acc.mean()), mcc=float(s.mcc.mean()), mcc_sd=float(s.mcc.std(ddof=1)) if len(s) > 1 else 0.0,
                        f1_macro=float(s.f1_macro.mean()), f1_macro_sd=float(s.f1_macro.std(ddof=1)) if len(s) > 1 else 0.0,
                        per_class_f1=pcf.mean(0).tolist(), n=len(s))
agg['multi'] = mt

# ---- zero-day
zd = {}
for f in sorted(glob.glob(f"{R}/ZD_*.json")):
    r = json.load(open(f)); zd[r['family']] = dict(n=r['n_fam'], **{m: v for m, v in r['models'].items()})
agg['zeroday'] = zd
json.dump(agg, open(OUT, 'w'), indent=1)

# ================= figures
cols = {'LR': '#9e9e9e', 'RF': '#8d6e63', 'LGBM': '#43a047', 'XGB_all': '#1e88e5', 'XGB_MI': '#90caf9', 'MLP': '#ffb74d',
        'CNN': '#fb8c00', 'H_insample': '#e53935', 'H_stack_MI': '#ba68c8', 'H_stack': '#6a1b9a'}
prots = [p for p in ['P1', 'P2', 'P3', 'P4'] if p in tab]
if prots:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, met, lab in [(axes[0], 'mcc', 'MCC'), (axes[1], 'f1_macro', 'Macro-F1')]:
        w = 0.08; x = np.arange(len(prots))
        for i, m in enumerate(ORDER):
            mu = [tab[p].get(m, {}).get(met, {}).get('mean', np.nan) for p in prots]
            sd = [tab[p].get(m, {}).get(met, {}).get('sd', 0) for p in prots]
            sd = [0 if p == 'P4' else v for p, v in zip(prots, sd)]
            ax.bar(x + (i - 4.5) * w, mu, w, yerr=sd, color=cols[m], label=NAMES[m], capsize=1.5, error_kw=dict(lw=0.6))
            if 'P4' in prots:
                pts = B[(B.protocol == 'P4') & (B.model == m)][met].values
                ax.scatter(np.full(len(pts), prots.index('P4') + (i - 4.5) * w), pts, s=4, color='k', zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(prots); ax.set_ylabel(lab); ax.set_ylim(0.0, 1.005); ax.grid(axis='y', alpha=0.3)
    axes[1].legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_protocol_mcc.png", dpi=300); plt.close()

if 'P4' in tab:
    s4 = B[B.protocol == 'P4']
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for m in ['XGB_all', 'LGBM', 'RF', 'XGB_MI', 'CNN', 'H_insample', 'H_stack']:
        s = s4[s4.model == m].sort_values('seed')
        ax.plot(s.seed + 1, s.mcc, marker='o', label=NAMES[m], color=cols[m], lw=1.2)
    ax.set_xlabel('Source-host-disjoint fold'); ax.set_ylabel('MCC'); ax.set_xticks(range(1, 6)); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_p4_folds.png", dpi=300); plt.close()

if mt:
    p = 'P4' if 'P4' in mt else list(mt)[0]
    ms_ = [m for m in ['LR', 'RF', 'LGBM', 'XGB_all', 'XGB_MI', 'MLP', 'CNN', 'H_insample', 'H_stack'] if m in mt[p]]
    H = np.array([mt[p][m]['per_class_f1'] for m in ms_])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(H, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=35, ha='right'); ax.set_yticks(range(len(ms_))); ax.set_yticklabels([NAMES[m] for m in ms_])
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i, j]:.2f}", ha='center', va='center', fontsize=6.5, color='w' if H[i, j] < 0.6 else 'k')
    plt.colorbar(im, ax=ax, fraction=0.03); ax.set_title(f'Per-class F1, ten-class task ({p})', fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_multiclass_heatmap.png", dpi=300); plt.close()

if zd:
    fams = sorted(zd, key=lambda k: -zd[k].get('XGB_all', {}).get('detection_rate', 0))
    ms_ = ['RF', 'LGBM', 'XGB_all', 'XGB_MI', 'CNN', 'H_stack']
    fig, ax = plt.subplots(figsize=(9, 3.4)); w = 0.13; x = np.arange(len(fams))
    for i, m in enumerate(ms_):
        ax.bar(x + (i - 2.5) * w, [zd[f][m]['detection_rate'] for f in fams], w, color=cols[m], label=NAMES[m])
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=20); ax.set_ylabel('Detection rate of unseen family'); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=3, frameon=False); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_zeroday.png", dpi=300); plt.close()

# MI ranking figure (P3 seed 0) and SHAP
k0 = 'P3_0' if 'P3_0' in mi else (list(mi)[0] if mi else None)
if k0:
    r = json.load(open(f"{R}/{k0.split('_')[0]}_bin_{k0.split('_')[1]}.json"))
    rk = pd.Series(r['mi']['ranking']); k = r['mi']['k_attributes']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]; ax.plot(range(1, len(rk) + 1), rk.values, 'o-', ms=3, color='#1e88e5')
    ax.axvline(k, ls='--', color='#e53935'); ax.text(k + 0.5, rk.max() * 0.9, f'knee: k = {k}', color='#e53935', fontsize=8)
    for i in range(min(k, 12)):
        ax.text(i + 1.3, rk.values[i], rk.index[i], fontsize=6.5, rotation=0, va='center')
    ax.set_xlabel('Attribute rank'); ax.set_ylabel('MI with label (nats, max over encoded columns)'); ax.grid(alpha=0.3)
    sh = pd.Series(r.get('shap_XGB_MI', {}))
    if len(sh):
        sh = sh.sort_values().tail(15); ax = axes[1]
        ax.barh(range(len(sh)), sh.values, color='#6a1b9a'); ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh.index, fontsize=7)
        ax.set_xlabel('mean |SHAP| (XGBoost on MI-selected features)'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_mi_shap.png", dpi=300); plt.close()

# CNN learning curve
if os.path.exists(f"{R}/P3_bin_0.json"):
    r = json.load(open(f"{R}/P3_bin_0.json")); h = r.get('cnn_history')
    if h:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(1, len(h['loss']) + 1), h['loss'], label='train'); ax.plot(range(1, len(h['val_loss']) + 1), h['val_loss'], label='validation')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Binary cross-entropy'); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{FIG}/fig_cnn_curve.png", dpi=300); plt.close()
print('aggregated', len(D), 'model-runs')
