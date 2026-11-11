"""Shared data preparation, protocols, feature selection and models.
Leakage-aware evaluation of MI-guided hybrid 1D-CNN + XGBoost on ToN-IoT (Train_Test_Network)."""
import os, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

DATASET = os.environ.get("DATASET", "ton")
LABELS = ['label', 'type']
if DATASET == 'edge':
    # Edge-IIoTset (ML subset). Identifier/payload columns dropped following the dataset authors' preprocessing.
    DATA = os.environ.get("EDGE_CSV", "data/edge_iiotset_ml.csv")
    IDS = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'http.file_data',
           'http.request.full_uri', 'icmp.transmit_timestamp', 'http.request.uri.query', 'tcp.options', 'tcp.payload',
           'tcp.srcport', 'tcp.dstport', 'udp.port', 'mqtt.msg', 'src_ip']
    CLASSES = ['normal', 'backdoor', 'ddos_http', 'ddos_icmp', 'ddos_tcp', 'ddos_udp', 'fingerprinting', 'mitm', 'password',
               'port_scanning', 'ransomware', 'sql_injection', 'uploading', 'vulnerability_scanner', 'xss']
else:
    DATA = os.environ.get("TON_CSV", "data/Train_Test_Network.csv")
    IDS = ['ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']
    CLASSES = ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss']
DROP_EXTRA = [c for c in os.environ.get("DROP_FEATURES", "").split(',') if c]


def load_raw():
    df = pd.read_csv(DATA, low_memory=False)
    if DATASET == 'edge':
        df = df.rename(columns={'Attack_label': 'label', 'Attack_type': 'type'})
        df['type'] = df['type'].astype(str).str.strip().str.lower()
        df['src_ip'] = df['ip.src_host'].astype(str)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith('str'):
            df[c] = df[c].astype(str)
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in IDS + LABELS + DROP_EXTRA]


def dedup_feature_space(df):
    """Remove exact duplicates in the model's feature space and vectors with conflicting labels."""
    feat = feature_cols(df)
    key = pd.util.hash_pandas_object(df[feat], index=False)
    n_types = df.groupby(key)['type'].transform('nunique')
    clean = df[n_types == 1].copy()
    clean['_key'] = key[n_types == 1].values
    clean = clean.drop_duplicates('_key').drop(columns='_key')
    return clean.reset_index(drop=True), int((n_types > 1).sum())


# ---------------------------------------------------------------- protocols
def split_protocol(df, protocol, seed, test_size=0.2):
    """Return (train_idx, test_idx) for one of four evaluation protocols.
    P1 random  : record-level stratified random split on raw data (common practice)
    P2 flow    : 5-tuple GroupShuffleSplit after record-level dedup (original manuscript)
    P3 dedup   : stratified random split after feature-space dedup
    P4 host    : StratifiedGroupKFold by source host after feature-space dedup (strict)"""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
    idx = np.arange(len(df))
    if protocol in ('P1', 'P3'):
        tr, te = train_test_split(idx, test_size=test_size, stratify=df['type'], random_state=seed)
    elif protocol == 'P2':
        g = df['src_ip'] + '|' + df['src_port'].astype(str) + '|' + df['dst_ip'] + '|' + df['dst_port'].astype(str) + '|' + df['proto']
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, groups=g))
    elif protocol == 'P6':
        # class-stratified chronological split: within every class, the earliest (1 - test_size) of the
        # flows (by timestamp) are used for training and the latest test_size for testing
        tr, te = [], []
        for _, g in df.groupby('type'):
            o = g.sort_values('ts', kind='mergesort').index.values
            cut = int(round(len(o) * (1 - test_size)))
            tr.extend(o[:cut]); te.extend(o[cut:])
        tr, te = np.sort(tr), np.sort(te)
    elif protocol == 'P4':
        # Fixed source-host -> fold map (generated once with StratifiedGroupKFold(5, shuffle=True, random_state=0),
        # scikit-learn 1.9). Older scikit-learn releases (e.g. 1.6) produce degenerate folds with shuffle=True,
        # so the map is shipped with the code to make P4 identical across environments.
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p4_host_folds.json')
        if os.path.exists(mp):
            fold = df['src_ip'].map(json.load(open(mp))).values
            assert not pd.isna(fold).any(), 'unknown source host in P4 fold map'
            te = idx[fold == seed % 5]; tr = idx[fold != seed % 5]
        else:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            folds = list(sgkf.split(idx, df['label'], groups=df['src_ip']))
            tr, te = folds[seed % 5]
    return np.asarray(tr), np.asarray(te)


def data_for_protocol(protocol):
    raw = load_raw()
    if protocol == 'P1':
        return raw
    if protocol == 'P2':
        return raw.drop_duplicates().reset_index(drop=True)
    d, _ = dedup_feature_space(raw)
    mn = int(os.environ.get('MIN_CLASS_N', '0'))
    if mn:
        vc = d['type'].value_counts(); d = d[d['type'].isin(vc[vc >= mn].index)].reset_index(drop=True)
    return d


# ---------------------------------------------------------------- encoding (fit on train only)
class Encoder:
    """Median imputation for numerics; one-hot for low-cardinality categoricals;
    train-frequency encoding for high-cardinality categoricals (dns_query, http_uri, ...)."""
    def __init__(self, max_ohe=20):
        self.max_ohe = max_ohe

    def fit(self, X):
        self.num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat = [c for c in X.columns if c not in self.num]
        self.med = X[self.num].median()
        self.ohe, self.freq = {}, {}
        for c in self.cat:
            vc = X[c].value_counts()
            if len(vc) <= self.max_ohe:
                self.ohe[c] = list(vc.index)
            else:
                self.freq[c] = (vc / len(X)).to_dict()
        return self

    def transform(self, X):
        out = {c: X[c].fillna(self.med[c]).astype(float).values for c in self.num}
        for c, levels in self.ohe.items():
            for lv in levels:
                out[f"{c}={lv}"] = (X[c].values == lv).astype(float)
        for c, m in self.freq.items():
            out[f"{c}_freq"] = X[c].map(m).fillna(0.0).astype(float).values
        F = pd.DataFrame(out)
        self.columns_ = list(F.columns)
        return F

    def parent(self, col):
        return col.split('=')[0].replace('_freq', '')


# ---------------------------------------------------------------- MI + geometric elbow
def mi_elbow(Xtr, ytr, enc, seed=42, n_sub=60000):
    """MI per encoded column (train only), aggregated to original attribute by max, then knee by
    maximum perpendicular distance to the chord (Kneedle-style). Returns selected encoded columns."""
    from sklearn.feature_selection import mutual_info_classif
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(Xtr), min(n_sub, len(Xtr)), replace=False)
    Xs = Xtr.iloc[sub]
    discrete = np.array([('=' in c) for c in Xs.columns])
    mi = mutual_info_classif(Xs.values, ytr[sub], discrete_features=discrete, random_state=seed)
    mi = pd.Series(mi, index=Xs.columns)
    parent = mi.groupby([enc.parent(c) for c in mi.index]).max().sort_values(ascending=False)
    y = parent.values; x = np.arange(1, len(y) + 1)
    num = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    d = num / np.sqrt((y[-1] - y[0]) ** 2 + (x[-1] - x[0]) ** 2)
    k = int(np.argmax(d)) + 1
    keep_parents = set(parent.index[:k])
    cols = [c for c in Xtr.columns if enc.parent(c) in keep_parents]
    return cols, parent, k


# ---------------------------------------------------------------- models
def xgb_model(n_classes, seed, spw=1.0):
    from xgboost import XGBClassifier
    kw = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=1, reg_lambda=1.0, tree_method='hist', n_jobs=2, random_state=seed, verbosity=0)
    if n_classes == 2:
        return XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=spw, **kw)
    return XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', **kw)


def build_cnn(k, n_classes, seed, l2=1e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers, Model
    tf.keras.utils.set_random_seed(seed)
    inp = layers.Input((k, 1))
    h = inp
    for f in (32, 64):
        h = layers.Conv1D(f, 3, padding='same', kernel_regularizer=regularizers.l2(l2))(h)
        h = layers.BatchNormalization()(h); h = layers.ReLU()(h)
        h = layers.MaxPooling1D(2, padding='same')(h)
    h = layers.Flatten()(h)
    h = layers.Dense(64, activation='relu')(h); h = layers.Dropout(0.3)(h)
    z = layers.Dense(32, activation='relu', name='deep_features')(h)
    out = layers.Dense(1, activation='sigmoid')(z) if n_classes == 2 else layers.Dense(n_classes, activation='softmax')(z)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy')
    return m


def fit_cnn(X, y, n_classes, seed, epochs=20, batch=1024, class_weight=None):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(np.arange(len(X)), test_size=0.1, random_state=seed, stratify=y)
    m = build_cnn(X.shape[1], n_classes, seed)
    cb = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5)]
    h = m.fit(X[tr, :, None], y[tr], validation_data=(X[va, :, None], y[va]), epochs=epochs, batch_size=batch,
              callbacks=cb, verbose=0, class_weight=class_weight)
    return m, h.history


def cnn_embed(m, X):
    import tensorflow as tf
    f = tf.keras.Model(m.input, m.get_layer('deep_features').output)
    return f.predict(X[:, :, None], batch_size=4096, verbose=0)


def cnn_proba(m, X):
    p = m.predict(X[:, :, None], batch_size=4096, verbose=0)
    return np.c_[1 - p[:, 0], p[:, 0]] if p.shape[1] == 1 else p


# ---------------------------------------------------------------- metrics
def metrics(y, proba, n_classes):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                                 matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix)
    pred = proba.argmax(1)
    r = dict(acc=accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred),
             f1_macro=f1_score(y, pred, average='macro'))
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        r.update(precision=precision_score(y, pred), recall=recall_score(y, pred), f1=f1_score(y, pred),
                 fpr=fp / (fp + tn), auc=roc_auc_score(y, proba[:, 1]), pr_auc=average_precision_score(y, proba[:, 1]),
                 tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    else:
        r['per_class_f1'] = f1_score(y, pred, average=None, labels=list(range(n_classes))).tolist()
        r['recall_macro'] = recall_score(y, pred, average='macro')
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}


# ---------------------------------------------------------------- v2 components (host-aware)
def host_folds(y, groups, n=3):
    """Stratified folds that keep every source host in a single fold (host-grouped cross-fitting).
    Falls back to ordinary stratified folds if a training part would miss a class."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    idx = np.arange(len(y))
    folds = list(StratifiedGroupKFold(n_splits=n).split(idx, y, groups=groups))
    if all(len(np.unique(y[a])) == len(np.unique(y)) for a, _ in folds):
        return folds, 'host'
    return list(StratifiedKFold(n, shuffle=True, random_state=0).split(idx, y)), 'record'


def mi_classwise_stable(Xtr, types, groups, enc, seed=42, n_sub=30000, n_groups=3):
    """Class-wise, host-stable MI selection. For every class (one-vs-rest) the MI of each encoded column is
    computed separately inside n_groups disjoint host groups; the stable score is the minimum over the groups
    in which the class occurs. Scores are aggregated to the original attribute by max, a geometric knee is taken
    per class, and the union of the per-class selections is returned."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import GroupKFold
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(Xtr), min(n_sub, len(Xtr)), replace=False)
    Xs, ts, gs = Xtr.iloc[sub], np.asarray(types)[sub], np.asarray(groups)[sub]
    discrete = np.array([('=' in c) for c in Xs.columns])
    parts = [b for _, b in GroupKFold(n_groups).split(Xs, groups=gs)]
    keep, per_class = set(), {}
    for c in np.unique(ts):
        scores = []
        for b in parts:
            yc = (ts[b] == c).astype(int)
            if yc.sum() < 5: continue
            scores.append(mutual_info_classif(Xs.values[b], yc, discrete_features=discrete, random_state=seed))
        if not scores: continue
        mi = pd.Series(np.min(scores, axis=0), index=Xs.columns)
        par = mi.groupby([enc.parent(col) for col in mi.index]).max().sort_values(ascending=False)
        yv = par.values; x = np.arange(1, len(yv) + 1)
        if len(yv) < 3 or yv[0] <= 0: continue
        d = np.abs((yv[-1] - yv[0]) * x - (x[-1] - x[0]) * yv + x[-1] * yv[0] - yv[-1] * x[0])
        k = int(np.argmax(d)) + 1
        per_class[str(c)] = list(par.index[:k]); keep |= set(par.index[:k])
    cols = [col for col in Xtr.columns if enc.parent(col) in keep]
    return cols, per_class


def oof_threshold(p_oof, y):
    """Decision threshold that maximises MCC on out-of-fold (held-out-host) training predictions."""
    from sklearn.metrics import matthews_corrcoef
    grid = np.unique(np.quantile(p_oof, np.linspace(0.005, 0.995, 199)))
    best = max(grid, key=lambda t: matthews_corrcoef(y, (p_oof >= t).astype(int)))
    return float(best)


def apply_threshold(proba, t):
    """Monotone remapping so that the decision at 0.5 equals the decision at threshold t (ranking unchanged)."""
    p = proba[:, 1]
    q = np.where(p < t, 0.5 * p / max(t, 1e-9), 0.5 + 0.5 * (p - t) / max(1 - t, 1e-9))
    return np.c_[1 - q, q]
