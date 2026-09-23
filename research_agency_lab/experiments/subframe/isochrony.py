import json, sys, numpy as np
sys.path.insert(0,'/tmp/subframe'); sys.path.insert(0,'.')
sys.path.insert(0,'research_agency_lab/experiments/learner_eval')
from occupancy import centroid_onsets
from app.rule_bind import ph_units
D='research_agency_lab/experiments/qaari_keys/modal_T300'
lay=json.load(open(f'{D}/layout.json')); C=lay['columns']
pl=[l for l in lay['levels'] if l['level']=='phonemes'][0]
vocab={t:i for i,t in enumerate(pl['vocab']) if len(t)==1}; blank=lay['blank']
tl=[l for l in lay['levels'] if l['level']=='tafkheem_or_taqeeq'][0]
idname={i:v for i,v in enumerate(tl['vocab'])}
V=set('َُِ'); MADD=set('اۥۦ')
sif={}
for line in open(f'{D}/sifat.jsonl'):
    x=json.loads(line); sif[x['id']]=x['levels']

def collect(speakers, nclips=400):
    per={'َ':[], 'ُ':[], 'ِ':[]}; weight={'heavy':[], 'light':[]}
    n=0
    for line in open(f'{D}/index.jsonl'):
        r=json.loads(line)
        if 'file' not in r or r.get('speaker') not in speakers: continue
        n+=1
        if n>nclips: break
        ids=sif.get(r['id'],{}).get('tafkheem_or_taqeeq',[])
        lp=np.fromfile(f"{D}/{r['file']}", dtype='<f4').reshape(r['frames'],C)[:, pl['first']:pl['first']+pl['width']]
        ph=r['ref_ph']; seq=[vocab[c] for c in ph]; units=ph_units(ph); nu=len(units)
        cen,_=centroid_onsets(lp,seq,blank)
        pos=np.array([cen[units[i][1]] for i in range(nu)])
        if np.any(~np.isfinite(pos)): continue
        dur=np.diff(pos, append=pos[-1]+1)
        har=[pos[min(i+2,nu-1)]-pos[i] for i in range(nu-1)
             if units[i][0] not in V and units[i][0] not in MADD and units[i+1][0] in V]
        har=[h for h in har if h>0]
        if len(har)<5: continue
        h=np.median(har)
        for i,(s,a,b) in enumerate(units):
            if s in per and dur[i]>0:
                per[s].append(dur[i]/h)
                if i>0 and a-1 < len(ids):
                    cls=idname.get(ids[units[i-1][1]],'')
                    if cls=='[مفخم]': weight['heavy'].append(dur[i]/h)
                    elif cls=='[مرقق]': weight['light'].append(dur[i]/h)
    return per, weight

def boot_ci(x, n=4000, rng=np.random.default_rng(7)):
    x=np.asarray(x); s=rng.choice(x,(n,len(x)),replace=True)
    m=np.median(s,axis=1); return np.percentile(m,[2.5,97.5])

ANCH={'Husary_Muallim_128kbps','Husary_128kbps','Husary_128kbps_Mujawwad'}
FAST={'Saood_ash-Shuraym_128kbps','MaherAlMuaiqly128kbps','Abdurrahmaan_As-Sudais_192kbps'}
for label,spk in (('ANCHORS',ANCH),('FAST IMAMS',FAST)):
    per,weight=collect(spk)
    print(f"\n=== {label} ===")
    meds={}
    for name,S in (('fatha','َ'),('damma','ُ'),('kasra','ِ')):
        v=per[S]; m=np.median(v); lo,hi=boot_ci(v); meds[name]=m
        print(f"  {name:6s} n={len(v):>5} median={m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    sp=max(meds.values())-min(meds.values())
    print(f"  ISOCHRONY spread = {sp:.4f} ({100*sp/np.mean(list(meds.values())):.2f}% of a vowel)")
    hv,lv=weight['heavy'],weight['light']
    if len(hv)>30 and len(lv)>30:
        mh,ml=np.median(hv),np.median(lv)
        lo,hi=boot_ci([a-b for a,b in zip(np.random.default_rng(3).choice(hv,2000),np.random.default_rng(4).choice(lv,2000))])
        print(f"  WEIGHT heavy n={len(hv)} median={mh:.4f} | light n={len(lv)} median={ml:.4f} | diff={mh-ml:+.4f}")
