import sys, json, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
df=C.data_for_protocol('P4'); Y,H,P=[],[],{}
for s in range(5):
    tr,te=C.split_protocol(df,'P4',s)
    for path in [f'results_pred/P4_bin_{s}_pred.npz', f'results_mlp/P4_bin_{s}_pred.npz']:
        z=np.load(path)
        for k in z.files:
            if k!='y': a=z[k]; P.setdefault(k,[]).append(a[:,1] if a.ndim==2 else a)
    Y.append(z['y']); H.append(df['src_ip'].values[te])
y=np.concatenate(Y); h=np.concatenate(H); pred={k:(np.concatenate(v)>=0.5).astype(int) for k,v in P.items() if len(v)==5}
hosts,inv=np.unique(h,return_inverse=True)
def counts(p):
    c=np.zeros((len(hosts),4))
    for j,(a,b) in enumerate([(0,0),(0,1),(1,0),(1,1)]): np.add.at(c[:,j],inv,((y==a)&(p==b)).astype(float))
    return c
def mcc(c):
    tn,fp,fn,tp=c; d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn-fp*fn)/d if d>0 else 0.0
Cn={k:counts(v) for k,v in pred.items()}
att=np.where(Cn['XGB_all'][:,2]+Cn['XGB_all'][:,3]>0)[0]; ben=np.setdiff1d(np.arange(len(hosts)),att)
rng=np.random.default_rng(0); W=np.zeros((2000,len(hosts)))
for b in range(2000): np.add.at(W[b],rng.choice(ben,len(ben)),1); np.add.at(W[b],rng.choice(att,len(att)),1)
out={}
for c,o in [('H_stack','CF_MLP'),('H_stack_MI','CF_MLP_MI'),('CF_MLP','XGB_all'),('CF_MLP','H_insample')]:
    d=np.array([mcc(W[b]@Cn[c])-mcc(W[b]@Cn[o]) for b in range(2000)]); pt=mcc(Cn[c].sum(0))-mcc(Cn[o].sum(0))
    p=min(1,2*min((d<=0).mean(),(d>=0).mean()))
    out[f'{c}_vs_{o}']=dict(diff=float(pt),ci=[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))],p=float(p),pooled=[float(mcc(Cn[c].sum(0))),float(mcc(Cn[o].sum(0)))])
    print(c,o,'pooled',round(mcc(Cn[c].sum(0)),3),round(mcc(Cn[o].sum(0)),3),'diff',round(pt,3),np.round(np.percentile(d,[2.5,97.5]),3),'p',round(p,3))
json.dump(out,open('mlp_frontend_boot.json','w'),indent=1)
