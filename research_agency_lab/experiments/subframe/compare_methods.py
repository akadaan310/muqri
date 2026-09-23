import json, sys, numpy as np
sys.path.insert(0,'/tmp/subframe'); sys.path.insert(0,'.')
sys.path.insert(0,'research_agency_lab/experiments/learner_eval')
from occupancy import centroid_onsets, peak_parabolic, expected_durations
from app.analysis import ctc_viterbi
from app.rule_bind import ph_units
D='research_agency_lab/experiments/qaari_keys/modal_T300'
lay=json.load(open(f'{D}/layout.json')); C=lay['columns']
pl=[l for l in lay['levels'] if l['level']=='phonemes'][0]
vocab={t:i for i,t in enumerate(pl['vocab']) if len(t)==1}; blank=lay['blank']
V=set('َُِ'); MADD=set('اۥۦ')

def vowel_stats(method, nclips=60, speaker='Husary_Muallim_128kbps'):
    per={'َ':[], 'ُ':[], 'ِ':[]}
    n=0
    for line in open(f'{D}/index.jsonl'):
        r=json.loads(line)
        if 'file' not in r or r.get('speaker')!=speaker: continue
        n+=1
        if n>nclips: break
        lp=np.fromfile(f"{D}/{r['file']}", dtype='<f4').reshape(r['frames'],C)[:, pl['first']:pl['first']+pl['width']]
        ph=r['ref_ph']; seq=[vocab[c] for c in ph]; units=ph_units(ph); nu=len(units)
        if method=='viterbi':
            _s,f,l=ctc_viterbi(lp,seq,blank); pos=np.array([f[units[i][1]] for i in range(nu)],float)
        elif method=='centroid':
            cen,_m=centroid_onsets(lp,seq,blank); pos=np.array([cen[units[i][1]] for i in range(nu)])
        else:
            pk=peak_parabolic(lp,seq,blank); pos=np.array([pk[units[i][1]] for i in range(nu)])
        if np.any(~np.isfinite(pos)): continue
        dur=np.diff(pos, append=pos[-1]+1)
        # haraka unit = consonant+vowel span
        har=[pos[min(i+2,nu-1)]-pos[i] for i in range(nu-1)
             if units[i][0] not in V and units[i][0] not in MADD and units[i+1][0] in V]
        har=[h for h in har if h>0]
        if len(har)<5: continue
        h=np.median(har)
        for i,(s,a,b) in enumerate(units):
            if s in per and dur[i]>0: per[s].append(dur[i]/h)
    return per

for method in ('viterbi','centroid','parabolic'):
    per=vowel_stats(method)
    med={k:np.median(v) for k,v in per.items() if v}
    uniq={k:len(set(np.round(v,4))) for k,v in per.items() if v}
    n={k:len(v) for k,v in per.items()}
    spread=max(med.values())-min(med.values())
    print(f"{method:11s} n={n} medians fatha={med.get('َ',0):.4f} damma={med.get('ُ',0):.4f} kasra={med.get('ِ',0):.4f}  SPREAD={spread:.4f}  uniq={uniq}")
