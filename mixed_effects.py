"""Mixed-effects analysis of paired MCC differences (random intercept for the protocol)."""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf
units = [('P2', s) for s in range(3)] + [('P3', s) for s in range(3)] + [('P4', f) for f in range(5)]
mcc = lambda p, s, m: json.load(open(f'results/{p}_bin_{s}.json'))['models'][m]['mcc']
for c in ['H_stack', 'H_stack_MI']:
    for o in ['H_insample', 'XGB_all', 'LGBM', 'RF']:
        d = pd.DataFrame([dict(prot=p, d=mcc(p, s, c) - mcc(p, s, o)) for p, s in units])
        f = smf.mixedlm('d ~ 1', d, groups=d['prot']).fit(reml=True)
        print(c, o, round(f.params['Intercept'], 4), round(f.bse['Intercept'], 4), round(f.pvalues['Intercept'], 3))
