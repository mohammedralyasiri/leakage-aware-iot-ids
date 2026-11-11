import sys, json, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
df = C.data_for_protocol('P4')
Y,H,P=[],[],{}
for s in range(5):
    tr,te=C.split_protocol(df,'P4',s); z=np.load(f'results_pred/P4_bin_{s}_pred.npz')
    Y.append(z['y']); H.append(df['src_ip'].values[te])
    for k in z.files:
        if k!='y': a=z[k]; P.setdefault(k,[]).append(a[:,1] if a.ndim==2 else a)
y=np.concatenate(Y); h=np.concatenate(H); pred={k:(np.concatenate(v)>=0.5).astype(int) for k,v in P.items()}
hosts,inv=np.unique(h,return_inverse=True)
def counts(p):
    c=np.zeros((len(hosts),4))
    for j,(a,b) in enumerate([(0,0),(0,1),(1,0),(1,1)]): np.add.at(c[:,j],inv,((y==a)&(p==b)).astype(float))
    return c
def mcc(c):
    tn,fp,fn,tp=c; d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn-fp*fn)/d if d>0 else 0.0
Cn={k:counts(v) for k,v in pred.items()}
pairs=[('H_stack_MI','XGB_all'),('H_stack','XGB_all'),('H_stack','LGBM'),('H_stack','H_insample'),('H_stack_MI','H_insample')]
att=np.where(Cn['XGB_all'][:,2]+Cn['XGB_all'][:,3]>0)[0]; ben=np.setdiff1d(np.arange(len(hosts)),att)
nflow=Cn['XGB_all'].sum(1); topben=ben[np.argsort(-nflow[ben])[:10]]
out={'n_attack_hosts':int(len(att)),'n_benign_hosts':int(len(ben))}
for c,o in pairs:
    full=mcc(Cn[c].sum(0))-mcc(Cn[o].sum(0))
    loo_att=[mcc(Cn[c].sum(0)-Cn[c][i])-mcc(Cn[o].sum(0)-Cn[o][i]) for i in att]
    loo_ben={hosts[i]:mcc(Cn[c].sum(0)-Cn[c][i])-mcc(Cn[o].sum(0)-Cn[o][i]) for i in topben}
    # sensitivity to number of attacker hosts: subsample k attacker hosts (all benign kept)
    rng=np.random.default_rng(1); sub={}
    for k in sorted({5,10,15,len(att)}):
        d=[]
        for _ in range(1000):
            keep=np.r_[ben, rng.choice(att,k,replace=False)]
            d.append(mcc(Cn[c][keep].sum(0))-mcc(Cn[o][keep].sum(0)))
        d=np.array(d); sub[k]=dict(mean=float(d.mean()),sd=float(d.std()),p5=float(np.percentile(d,5)),p95=float(np.percentile(d,95)),frac_pos=float((d>0).mean()))
    out[f'{c}_vs_{o}']=dict(full=full,loo_attack_min=float(min(loo_att)),loo_attack_max=float(max(loo_att)),
                             loo_benign_top={k:float(v) for k,v in loo_ben.items()},subsample=sub)
    big=hosts[topben[0]]
    print(f'{c} vs {o}: full {full:+.3f} | LOO attacker [{min(loo_att):+.3f},{max(loo_att):+.3f}] | drop {big}: {loo_ben[big]:+.3f} | LOO top-10 benign [{min(loo_ben.values()):+.3f},{max(loo_ben.values()):+.3f}]')
    for k,v in sub.items(): print(f'   k={k:2d} mean {v["mean"]:+.3f} sd {v["sd"]:.3f} 90%[{v["p5"]:+.3f},{v["p95"]:+.3f}] P(>0)={v["frac_pos"]:.2f}')
print('largest benign host',hosts[topben[0]], int(nflow[topben[0]]),'flows; share of benign test flows', round(nflow[topben[0]]/nflow[ben].sum(),3))
json.dump(out,open('host_sensitivity.json','w'),indent=1)
i195=int(np.where(hosts=='192.168.1.195')[0][0]); res195={}
for c,o in pairs:
    d=mcc(Cn[c].sum(0)-Cn[c][i195])-mcc(Cn[o].sum(0)-Cn[o][i195]); res195[f'{c}_vs_{o}']=float(d)
    print('drop .195', c, o, f'{d:+.3f}', 'pooled MCC without', round(mcc(Cn[c].sum(0)-Cn[c][i195]),3), round(mcc(Cn[o].sum(0)-Cn[o][i195]),3))
out['drop_195']=res195; json.dump(out,open('host_sensitivity.json','w'),indent=1)
