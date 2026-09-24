"""Audio quality per reciter source (SYNTHESIS.md): effective bandwidth and speech-to-floor range on the same 6 ayahs."""
import io, sys, urllib.request, numpy as np, librosa, json
from concurrent.futures import ThreadPoolExecutor
F=['Husary_128kbps','Husary_Muallim_128kbps','Husary_128kbps_Mujawwad','Abdurrahmaan_As-Sudais_192kbps',
   'Minshawy_Murattal_128kbps','Minshawy_Mujawwad_192kbps','Abdul_Basit_Murattal_192kbps','Alafasy_128kbps',
   'MaherAlMuaiqly128kbps','Yasser_Ad-Dussary_128kbps','Muhammad_Ayyoub_128kbps','Hudhaify_128kbps',
   'Saood_ash-Shuraym_128kbps','Hani_Rifai_192kbps','Muhammad_Jibreel_128kbps','Nasser_Alqatami_128kbps']
V=[(2,2),(2,255),(18,10),(36,1),(67,2),(112,1)]
def load(f,s,a):
    d=urllib.request.urlopen(urllib.request.Request(f'https://everyayah.com/data/{f}/{s:03d}{a:03d}.mp3',headers={'User-Agent':'q'}),timeout=60).read()
    return librosa.load(io.BytesIO(d),sr=None,mono=True)
def metrics(f):
    bw=[];nf=[];hf=[];srs=set()
    for s,a in V:
        try: y,sr=load(f,s,a)
        except Exception as e: continue
        srs.add(sr)
        S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512))**2
        e=S.sum(0); speech=S[:,e>np.percentile(e,70)].mean(1)
        fr=librosa.fft_frequencies(sr=sr,n_fft=2048)
        db=10*np.log10(speech+1e-20); pk=db.max()
        # effective bandwidth: highest frequency whose smoothed speech PSD is within 60 dB of the peak
        sm=np.convolve(db,np.ones(9)/9,'same'); bw.append(fr[np.where(sm>pk-60)[0].max()])
        fe=10*np.log10(e+1e-20); nf.append(np.percentile(fe,95)-np.percentile(fe,5))   # speech-to-floor range, dB
        hf.append(10*np.log10(speech[fr>=5000].sum()/speech.sum()))                     # energy above 5 kHz, dB
    return f,sorted(srs),round(float(np.median(bw))),round(float(np.median(nf)),1),round(float(np.median(hf)),1)
with ThreadPoolExecutor(8) as p:
    rows=list(p.map(metrics,F))
print(f"{'source':34s} sr      bw60dB_Hz  speech-floor_dB  >5kHz_dB")
for r in sorted(rows,key=lambda r:-r[2]): print(f"{r[0]:34s} {str(r[1]):8s} {r[2]:9d}  {r[3]:10.1f}  {r[4]:8.1f}")
json.dump(rows,open('research_agency_lab/experiments/external/voice_quality.json','w'))
