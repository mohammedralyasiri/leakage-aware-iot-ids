"""One experimental run: python run.py <protocol P1..P4> <task bin|multi> <seed> [--models a,b]"""
import sys, os, json, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

protocol, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
ALL = ['LR', 'RF', 'LGBM', 'XGB_all', 'XGB_MI', 'MLP', 'CNN', 'H_insample', 'H_stack_MI', 'H_stack']
V2 = ['XGB_thr', 'XGB_cs', 'CF_H', 'CF_H_thr', 'CF_H_cs', 'CF_v2']
models = sys.argv[5].split(',') if len(sys.argv) > 5 and sys.argv[4] == '--models' else ALL
OUT = os.path.join(os.environ.get("RESULTS_DIR", "results"), f"{protocol}_{task}_{seed}.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = data_for_protocol(protocol)
tr, te = split_protocol(df, protocol, seed)
if task == 'bin':
    y = df['label'].values.astype(int); K = 2
else:
    present = [c for c in CLASSES if c in set(df['type'])]
    y = df['type'].map({c: i for i, c in enumerate(present)}).values.astype(int); K = len(present)
feat = feature_cols(df)
enc = Encoder().fit(df.iloc[tr][feat])
Xtr = enc.transform(df.iloc[tr][feat]); Xte = enc.transform(df.iloc[te][feat])
ytr, yte = y[tr], y[te]
res = dict(classes=(present if task != 'bin' else ['normal', 'attack']), protocol=protocol, task=task, seed=seed, n=len(df), n_train=len(tr), n_test=len(te),
           test_class_counts=pd.Series(yte).value_counts().sort_index().to_dict(), n_encoded=Xtr.shape[1], models={})
if protocol == 'P4':
    res['test_attack_hosts'] = sorted(df.iloc[te].loc[df.iloc[te]['label'] == 1, 'src_ip'].unique().tolist())
    res['train_attack_hosts'] = sorted(df.iloc[tr].loc[df.iloc[tr]['label'] == 1, 'src_ip'].unique().tolist())

t0 = time.time()
sel, mi_rank, k = mi_elbow(Xtr, ytr, enc, seed=seed)
res['mi'] = dict(k_attributes=k, n_columns=len(sel), ranking=mi_rank.round(4).to_dict(), time=time.time() - t0)
slog = lambda a: np.sign(a) * np.log1p(np.abs(a))   # tames heavy-tailed byte/packet counts for neural nets
sc_all = StandardScaler().fit(slog(Xtr.values)); sc_mi = StandardScaler().fit(slog(Xtr[sel].values))
A_tr, A_te = sc_all.transform(slog(Xtr.values)), sc_all.transform(slog(Xte.values))
M_tr, M_te = sc_mi.transform(slog(Xtr[sel].values)), sc_mi.transform(slog(Xte[sel].values))
if os.environ.get('PERM_SEED'):
    perm = np.random.RandomState(int(os.environ['PERM_SEED'])).permutation(M_tr.shape[1])
    M_tr, M_te = M_tr[:, perm], M_te[:, perm]; res['cnn_input_permutation'] = perm.tolist()
spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1) if K == 2 else 1.0
res['scale_pos_weight'] = float(spw)


PRED = {}
def record(name, fit_time, proba, infer_time, size_bytes, extra=None):
    if os.environ.get('SAVE_PRED'): PRED[name] = proba.astype(np.float32)
    m = metrics(yte, proba, K)
    m.update(train_s=fit_time, infer_us_per_flow=1e6 * infer_time / len(yte), size_kb=size_bytes / 1024)
    if extra: m.update(extra)
    res['models'][name] = m
    print(f"{protocol} {task} s{seed} {name:11s} acc={m['acc']:.4f} f1m={m['f1_macro']:.4f} mcc={m['mcc']:.4f} t={fit_time:.0f}s", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)


def timed(fn):
    t = time.time(); out = fn(); return out, time.time() - t


def sz(obj):
    return len(pickle.dumps(obj))


def xgb_size(m):
    return len(m.get_booster().save_raw('json'))


if 'LR' in models:
    from sklearn.linear_model import LogisticRegression
    m, ft = timed(lambda: LogisticRegression(max_iter=300, n_jobs=2).fit(A_tr, ytr))
    p, it = timed(lambda: m.predict_proba(A_te)); record('LR', ft, p, it, sz(m))
if 'RF' in models:
    from sklearn.ensemble import RandomForestClassifier
    m, ft = timed(lambda: RandomForestClassifier(100, n_jobs=2, random_state=seed, min_samples_leaf=1).fit(Xtr.values, ytr))
    p, it = timed(lambda: m.predict_proba(Xte.values)); record('RF', ft, p, it, sz(m))
if 'LGBM' in models:
    from lightgbm import LGBMClassifier
    # multi-class: the shared configuration diverges; use a more conservative one (see manuscript Section 4.1)
    lkw = dict(learning_rate=0.1, num_leaves=63) if K == 2 else dict(learning_rate=0.05, num_leaves=31, min_child_samples=20, min_sum_hessian_in_leaf=1.0)
    m, ft = timed(lambda: LGBMClassifier(n_estimators=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                         n_jobs=2, random_state=seed, verbose=-1, **lkw).fit(Xtr.values, ytr))
    p, it = timed(lambda: m.predict_proba(Xte.values)); record('LGBM', ft, p, it, len(m.booster_.model_to_string()))
if 'XGB_all' in models:
    m, ft = timed(lambda: xgb_model(K, seed, spw).fit(Xtr.values, ytr))
    p, it = timed(lambda: m.predict_proba(Xte.values)); record('XGB_all', ft, p, it, xgb_size(m))
if 'XGB_MI' in models:
    m, ft = timed(lambda: xgb_model(K, seed, spw).fit(Xtr[sel].values, ytr))
    p, it = timed(lambda: m.predict_proba(Xte[sel].values)); record('XGB_MI', ft, p, it, xgb_size(m))
    try:
        import shap
        ex = shap.TreeExplainer(m); sub = np.random.RandomState(seed).choice(len(Xte), min(3000, len(Xte)), replace=False)
        sv = ex.shap_values(Xte[sel].values[sub])
        sv = np.abs(np.array(sv))
        sv = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)
        res['shap_XGB_MI'] = dict(zip(sel, sv.tolist()))
    except Exception as e:
        res['shap_err'] = str(e)
if 'MLP' in models:
    import tensorflow as tf
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(seed)
    mlp = tf.keras.Sequential([layers.Input((M_tr.shape[1],)), layers.Dense(128, activation='relu'), layers.Dropout(0.3),
                               layers.Dense(64, activation='relu'), layers.Dense(1 if K == 2 else K, activation='sigmoid' if K == 2 else 'softmax')])
    mlp.compile('adam', 'binary_crossentropy' if K == 2 else 'sparse_categorical_crossentropy')
    cb = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
    _, ft = timed(lambda: mlp.fit(M_tr, ytr, validation_split=0.1, epochs=25, batch_size=512, callbacks=cb, verbose=0))
    p, it = timed(lambda: mlp.predict(M_te, batch_size=4096, verbose=0))
    p = np.c_[1 - p[:, 0], p[:, 0]] if K == 2 else p
    record('MLP', ft, p, it, mlp.count_params() * 4)

need_cnn = any(x in models for x in ['CNN', 'H_insample', 'H_stack_MI', 'H_stack'])
if need_cnn:
    (cnn, hist), ft_cnn = timed(lambda: fit_cnn(M_tr, ytr, K, seed))
    res['cnn_history'] = {k: [float(v) for v in vals] for k, vals in hist.items() if k in ('loss', 'val_loss')}
    cnn_bytes = cnn.count_params() * 4
    p, it = timed(lambda: cnn_proba(cnn, M_te)); record('CNN', ft_cnn, p, it, cnn_bytes, dict(epochs=len(hist['loss'])))
    (Z_tr_in, Z_te), it_emb = timed(lambda: (cnn_embed(cnn, M_tr), cnn_embed(cnn, M_te)))
    it_emb_te = it_emb * len(M_te) / (len(M_tr) + len(M_te))
    if 'H_insample' in models:
        m, ft = timed(lambda: xgb_model(K, seed, spw).fit(Z_tr_in, ytr))
        p, it = timed(lambda: m.predict_proba(Z_te)); record('H_insample', ft + ft_cnn, p, it + it_emb_te, xgb_size(m) + cnn_bytes)
    if 'H_stack_MI' in models or 'H_stack' in models:
        # cross-fitted CNN posteriors: out-of-fold on train, fold-average on test (no label leakage into the meta-learner)
        P_oof = np.zeros((len(M_tr), K)); P_te = np.zeros((len(M_te), K)); t = time.time()
        for f, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(M_tr, ytr)):
            cf, _ = fit_cnn(M_tr[a], ytr[a], K, seed + 100 + f)
            P_oof[b] = cnn_proba(cf, M_tr[b]); P_te += cnn_proba(cf, M_te) / 3
        ft_oof = time.time() - t
        P_oof, P_te = P_oof[:, 1:], P_te[:, 1:]  # drop redundant column
        for name, Xa, Xb in [('H_stack_MI', Xtr[sel].values, Xte[sel].values), ('H_stack', Xtr.values, Xte.values)]:
            if name not in models: continue
            m, ft = timed(lambda: xgb_model(K, seed, spw).fit(np.c_[Xa, P_oof], ytr))
            p, it = timed(lambda: m.predict_proba(np.c_[Xb, P_te]))
            imp = m.get_booster().get_score(importance_type='gain'); tot = sum(imp.values())
            share = sum(v for k2, v in imp.items() if int(k2[1:]) >= Xa.shape[1]) / tot
            record(name, ft + ft_oof, p, it + it_emb_te * 3, xgb_size(m) + 3 * cnn_bytes, dict(gain_share_cnn=share))


# ---------------------------------------------------------------- cross-fitted stacking with an MLP front-end (architecture control)
if 'CF_MLP' in models or 'CF_MLP_MI' in models:
    import tensorflow as tf
    from tensorflow.keras import layers
    def fit_mlp_(Xa, ya, s):
        tf.keras.utils.set_random_seed(s)
        m = tf.keras.Sequential([layers.Input((Xa.shape[1],)), layers.Dense(128, activation='relu'), layers.Dropout(0.3),
                                 layers.Dense(64, activation='relu'), layers.Dense(1 if K == 2 else K, activation='sigmoid' if K == 2 else 'softmax')])
        m.compile('adam', 'binary_crossentropy' if K == 2 else 'sparse_categorical_crossentropy')
        m.fit(Xa, ya, validation_split=0.1, epochs=25, batch_size=512, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
        return m
    def mp(m, X):
        p = m.predict(X, batch_size=4096, verbose=0); return np.c_[1 - p[:, 0], p[:, 0]] if K == 2 else p
    P_oof = np.zeros((len(M_tr), K)); P_te = np.zeros((len(M_te), K)); t = time.time()
    for f, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(M_tr, ytr)):
        cf = fit_mlp_(M_tr[a], ytr[a], seed + 100 + f); P_oof[b] = mp(cf, M_tr[b]); P_te += mp(cf, M_te) / 3
    ft_oof = time.time() - t; P_oof, P_te = P_oof[:, 1:], P_te[:, 1:]
    for name, Xa, Xb in [('CF_MLP_MI', Xtr[sel].values, Xte[sel].values), ('CF_MLP', Xtr.values, Xte.values)]:
        if name not in models: continue
        m, ft = timed(lambda: xgb_model(K, seed, spw).fit(np.c_[Xa, P_oof], ytr))
        p, it = timed(lambda: m.predict_proba(np.c_[Xb, P_te]))
        imp = m.get_booster().get_score(importance_type='gain'); tot = sum(imp.values())
        share = sum(v for k2, v in imp.items() if int(k2[1:]) >= Xa.shape[1]) / tot
        record(name, ft + ft_oof, p, it, xgb_size(m), dict(gain_share_mlp=share))

# ---------------------------------------------------------------- v2: host-aware cross-fitted stacking + ablation
if any(x in models for x in V2):
    g_tr = df.iloc[tr]['src_ip'].astype(str).values
    folds, fold_kind = host_folds(ytr, g_tr, 3)
    res['v2'] = dict(fold_kind=fold_kind)
    t = time.time()
    sel2, per_class = mi_classwise_stable(Xtr, df.iloc[tr]['type'].values, g_tr, enc, seed=seed)
    res['v2'].update(cs_columns=len(sel2), cs_attributes=sorted({enc.parent(c) for c in sel2}), cs_time=time.time() - t)
    sc2 = StandardScaler().fit(slog(Xtr[sel2].values))
    C_tr, C_te = sc2.transform(slog(Xtr[sel2].values)), sc2.transform(slog(Xte[sel2].values))

    def meta_oof(Xa, extra=None):
        """host-grouped OOF probabilities of the XGBoost (meta-)learner on the training set (binary only)"""
        p = np.zeros(len(ytr))
        for a, b in folds:
            Za = Xa[a] if extra is None else np.c_[Xa[a], extra[a]]
            Zb = Xa[b] if extra is None else np.c_[Xa[b], extra[b]]
            p[b] = xgb_model(K, seed, spw).fit(Za, ytr[a]).predict_proba(Zb)[:, 1]
        return p

    if 'XGB_thr' in models and K == 2:
        t = time.time(); m = xgb_model(K, seed, spw).fit(Xtr.values, ytr); thr = oof_threshold(meta_oof(Xtr.values), ytr); ft = time.time() - t
        p, it = timed(lambda: m.predict_proba(Xte.values)); record('XGB_thr', ft, apply_threshold(p, thr), it, xgb_size(m), dict(threshold=thr))
    if 'XGB_cs' in models:
        m, ft = timed(lambda: xgb_model(K, seed, spw).fit(Xtr[sel2].values, ytr))
        p, it = timed(lambda: m.predict_proba(Xte[sel2].values)); record('XGB_cs', ft, p, it, xgb_size(m))

    def host_posteriors(Min, Mte, tag):
        P_oof = np.zeros((len(Min), K)); P_te = np.zeros((len(Mte), K)); t = time.time()
        for f, (a, b) in enumerate(folds):
            cf, _ = fit_cnn(Min[a], ytr[a], K, seed + 200 + f)
            P_oof[b] = cnn_proba(cf, Min[b]); P_te += cnn_proba(cf, Mte) / len(folds)
        return P_oof[:, 1:], P_te[:, 1:], time.time() - t

    def stack(name, Po, Pt, fto, thr=False):
        m, ft = timed(lambda: xgb_model(K, seed, spw).fit(np.c_[Xtr.values, Po], ytr))
        p, it = timed(lambda: m.predict_proba(np.c_[Xte.values, Pt]))
        imp = m.get_booster().get_score(importance_type='gain'); tot = sum(imp.values())
        share = sum(v for k2, v in imp.items() if int(k2[1:]) >= Xtr.shape[1]) / tot
        extra = dict(gain_share_cnn=share)
        if thr and K == 2:
            t = time.time(); th = oof_threshold(meta_oof(Xtr.values, Po), ytr); ft += time.time() - t
            p = apply_threshold(p, th); extra['threshold'] = th
        record(name, ft + fto, p, it, xgb_size(m) + 3 * cnn_bytes if 'cnn_bytes' in globals() else xgb_size(m), extra)

    if 'CF_H' in models or 'CF_H_thr' in models:
        Po, Pt, fto = host_posteriors(M_tr, M_te, 'mi')
        if 'CF_H' in models: stack('CF_H', Po, Pt, fto)
        if 'CF_H_thr' in models: stack('CF_H_thr', Po, Pt, fto, thr=True)
    if 'CF_H_cs' in models or 'CF_v2' in models:
        Po, Pt, fto = host_posteriors(C_tr, C_te, 'cs')
        if 'CF_H_cs' in models: stack('CF_H_cs', Po, Pt, fto)
        if 'CF_v2' in models: stack('CF_v2', Po, Pt, fto, thr=True)
res['done'] = True
json.dump(res, open(OUT, 'w'), indent=1)
if os.environ.get('SAVE_PRED'):
    np.savez_compressed(OUT.replace('.json', '_pred.npz'), y=yte, **PRED)
