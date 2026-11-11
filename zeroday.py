"""Leave-one-attack-family-out (unseen attack) evaluation on the feature-space-deduplicated data.
python zeroday.py <family> <seed>"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

fam, seed = sys.argv[1], int(sys.argv[2])
OUT = os.path.join(os.environ.get("RESULTS_DIR", "results"), f"ZD_{fam}_{seed}.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df, _ = dedup_feature_space(load_raw())
_mn = int(os.environ.get('MIN_CLASS_N', '0'))
if _mn:
    _vc = df['type'].value_counts(); df = df[df['type'].isin(_vc[_vc >= _mn].index)].reset_index(drop=True)
idx = np.arange(len(df)); is_fam = (df['type'] == fam).values
normal = idx[df['type'].values == 'normal']; others = idx[(~is_fam) & (df['type'].values != 'normal')]
n_tr, n_te = train_test_split(normal, test_size=0.2, random_state=seed)
o_tr, _ = train_test_split(others, test_size=0.2, random_state=seed, stratify=df['type'].values[others])
tr = np.r_[n_tr, o_tr]; te = np.r_[n_te, idx[is_fam]]
y = df['label'].values.astype(int); feat = feature_cols(df)
enc = Encoder().fit(df.iloc[tr][feat]); Xtr = enc.transform(df.iloc[tr][feat]); Xte = enc.transform(df.iloc[te][feat])
ytr, yte = y[tr], y[te]
sel, _, k = mi_elbow(Xtr, ytr, enc, seed=seed)
slog = lambda a: np.sign(a) * np.log1p(np.abs(a))
sc = StandardScaler().fit(slog(Xtr[sel].values)); M_tr, M_te = sc.transform(slog(Xtr[sel].values)), sc.transform(slog(Xte[sel].values))
spw = (ytr == 0).sum() / (ytr == 1).sum()
res = dict(family=fam, seed=seed, n_fam=int(is_fam.sum()), n_test_normal=len(n_te), k=k, models={})


def rec(name, p):
    pred = p.argmax(1); a = yte == 1
    res['models'][name] = dict(detection_rate=float(pred[a].mean()), fpr=float(pred[~a].mean()),
                               auc=float(__import__('sklearn.metrics', fromlist=['x']).roc_auc_score(yte, p[:, 1])))
    print(fam, name, res['models'][name], flush=True)


from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
rec('RF', RandomForestClassifier(100, n_jobs=2, random_state=seed).fit(Xtr.values, ytr).predict_proba(Xte.values))
rec('LGBM', LGBMClassifier(n_estimators=300, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           n_jobs=2, random_state=seed, verbose=-1).fit(Xtr.values, ytr).predict_proba(Xte.values))
rec('XGB_all', xgb_model(2, seed, spw).fit(Xtr.values, ytr).predict_proba(Xte.values))
rec('XGB_MI', xgb_model(2, seed, spw).fit(Xtr[sel].values, ytr).predict_proba(Xte[sel].values))
if os.environ.get('CS_ONLY'):
    sel2, _ = mi_classwise_stable(Xtr, df.iloc[tr]['type'].values, df.iloc[tr]['src_ip'].astype(str).values, enc, seed=seed)
    res['cs_attributes'] = sorted({enc.parent(c) for c in sel2})
    rec('XGB_cs', xgb_model(2, seed, spw).fit(Xtr[sel2].values, ytr).predict_proba(Xte[sel2].values))
    json.dump(res, open(OUT, 'w'), indent=1); sys.exit(0)
if os.environ.get('NO_CNN'):
    json.dump(res, open(OUT, 'w'), indent=1); sys.exit(0)
cnn, _ = fit_cnn(M_tr, ytr, 2, seed); rec('CNN', cnn_proba(cnn, M_te))
if os.environ.get('FAST'):
    json.dump(res, open(OUT, 'w'), indent=1); sys.exit(0)
P_oof = np.zeros(len(M_tr)); P_te = np.zeros(len(M_te))
for f, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(M_tr, ytr)):
    cf, _ = fit_cnn(M_tr[a], ytr[a], 2, seed + 100 + f); P_oof[b] = cnn_proba(cf, M_tr[b])[:, 1]; P_te += cnn_proba(cf, M_te)[:, 1] / 3
rec('H_stack', xgb_model(2, seed, spw).fit(np.c_[Xtr.values, P_oof], ytr).predict_proba(np.c_[Xte.values, P_te]))
json.dump(res, open(OUT, 'w'), indent=1)
