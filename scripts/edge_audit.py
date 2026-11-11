import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import common as C
df=C.load_raw()
print(len(df), df['type'].value_counts().to_dict())
for drop in [[],['tcp.checksum','icmp.checksum','tcp.ack_raw','udp.stream']]:
    C.DROP_EXTRA=drop
    fc=C.feature_cols(df)
    h=pd.util.hash_pandas_object(df[fc].astype(str),index=False)
    g=df.assign(h=h.values)
    nt=g.groupby('h')['type'].nunique()
    conf=g['h'].isin(nt[nt>1].index).sum()
    print(len(fc),'unique',h.nunique(),'dup%',round(100*(1-h.nunique()/len(df)),1),'conflict rows',conf)
print('hosts benign',df.loc[df.label==0,'src_ip'].nunique(),'attack',df.loc[df.label==1,'src_ip'].nunique())
