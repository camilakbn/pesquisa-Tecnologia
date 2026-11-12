import numpy as np, json
N_BINS=60; DAYS=90; SEEDS=30; FIXED_PERIOD=3; THRESH=0.75
NOISE=0.02; LOSS=0.05
def route_len(pts):
    # nearest neighbour + 2-opt, depot at (0,0)
    if len(pts)==0: return 0.0
    P=np.vstack([[0,0],pts]); n=len(P)
    D=np.linalg.norm(P[:,None]-P[None],axis=2)
    tour=[0]; left=set(range(1,n))
    while left:
        j=min(left,key=lambda k:D[tour[-1],k]); tour.append(j); left.remove(j)
    tour.append(0)
    improved=True
    while improved:
        improved=False
        for i in range(1,len(tour)-2):
            for k in range(i+1,len(tour)-1):
                a,b,c,d=tour[i-1],tour[i],tour[k],tour[k+1]
                if D[a,c]+D[b,d] < D[a,b]+D[c,d]-1e-9:
                    tour[i:k+1]=tour[i:k+1][::-1]; improved=True
    return sum(D[tour[i],tour[i+1]] for i in range(len(tour)-1))*1.3  # fator de tortuosidade urbana
def run(seed):
    rng=np.random.default_rng(seed)
    pos=rng.uniform(-5,5,(N_BINS,2))
    base=rng.lognormal(np.log(0.15),0.45,N_BINS)  # taxa media diaria de enchimento
    res={}
    for strat in ['fixa','iot','iot2']:
        r2=np.random.default_rng(seed+1000)
        lvl=np.zeros(N_BINS); est=np.zeros(N_BINS); hist=[[] for _ in range(N_BINS)]
        km=0; trips=0; visits=0; over=0; fills=[]
        for d in range(DAYS):
            inc=np.clip(base*r2.normal(1,0.35,N_BINS),0,None)
            lvl=lvl+inc
            over+=int((lvl>1).sum()); lvl=np.minimum(lvl,1)
            if strat=='fixa':
                sel=np.arange(N_BINS) if d%FIXED_PERIOD==FIXED_PERIOD-1 else np.array([],int)
            else:
                meas=np.clip(lvl+r2.normal(0,NOISE,N_BINS),0,1)
                ok=r2.random(N_BINS)>LOSS
                for i in range(N_BINS):
                    if ok[i]: hist[i].append(meas[i]-est[i] if len(hist[i]) else meas[i]); est[i]=meas[i]
                    else:
                        rate=np.mean(hist[i][-5:]) if hist[i] else 0.15; est[i]=min(1,est[i]+max(rate,0))
                rate=np.array([np.mean(h[-5:]) if h else 0.15 for h in hist])
                crit=(est+1.2*rate>=1.0)
                if strat=='iot':
                    sel=np.where((est>=THRESH)|crit)[0]
                else:
                    sel=np.where(est+FIXED_PERIOD*1.15*rate>=0.95)[0] if d%FIXED_PERIOD==FIXED_PERIOD-1 else np.array([],int)
            if len(sel):
                km+=route_len(pos[sel]); trips+=1; visits+=len(sel); fills+=list(lvl[sel])
                lvl[sel]=0; est[sel]=0
        res[strat]=dict(km=km,visits=visits,over=over,fill=float(np.mean(fills)),trips=trips)
    return res
R=[run(s) for s in range(SEEDS)]
out={}
for st in ['fixa','iot','iot2']:
    for m in ['km','visits','over','fill','trips']:
        v=np.array([r[st][m] for r in R]); out[f'{st}_{m}']=(round(float(v.mean()),3),round(float(v.std(ddof=1)),3))
from math import sqrt
for st in ['iot','iot2']:
    for m in ['km','visits','over','trips']:
        red=np.array([1-r[st][m]/r['fixa'][m] for r in R]); out[f'red_{st}_{m}']=(round(float(red.mean()),4),round(float(red.std(ddof=1)),4))
    for m in ['km','over']:
        d=np.array([r['fixa'][m]-r[st][m] for r in R]); out[f't_{st}_{m}']=round(float(d.mean()/(d.std(ddof=1)/sqrt(SEEDS))),2)
json.dump(out,open('/tmp/poc/res.json','w'),indent=1); print(json.dumps(out))
