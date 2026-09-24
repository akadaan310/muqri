"""Upload page: drop a recitation in, get the full mastery report.

This is the engine's evaluation surface. Everything measured so far comes from the EveryAyah corpus;
this is where a real recording, recorded by a real person, goes through the same path.

    .venv/bin/python -m app.webapp            # serves on 0.0.0.0:8088

The acoustic model is 2.4 GB and takes ~40 s to load, so it is warmed at startup rather than on the
first upload — otherwise the first request looks broken. `GET /health` reports whether it is warm.

Inference is local CPU (4 cores, no GPU).

A warning about latency numbers measured on this box: it is shared. At the time of writing the load
average was 43 on 4 cores — three Julia jobs and two Python jobs from an unrelated project — so the
engine measured RTF ~2 per verse and adding threads made it *slower* (RTF 3.65 at one thread, 22.79
at four), which is thread thrashing under ~10x oversubscription, not a property of the model. The
same model measured RTF 0.52 when the box was quiet. Profile again on an idle box before drawing any
conclusion about speed; the analysis itself is free either way (0.26 s per verse, RTF 0.02) and
essentially all the time is in the acoustic model.
"""

# No ``from __future__ import annotations`` here. FastAPI resolves endpoint annotations at runtime,
# and the deferred form leaves ``UploadFile`` as an unresolved ForwardRef, which pydantic rejects with
# "is not fully defined". app/api.py carries the same warning; it cost a debugging round to rediscover.

import io
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PORT = 8088
# every protocol take is kept: the audio, the engine's full report and the scorecard (gitignored --
# these are a person's voice)
PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "research_agency_lab/experiments/protocol_recordings"
# a learner's accumulated evidence, one file per learner id (gitignored: personal)
LEARNERS_DIR = Path(__file__).resolve().parents[1] / "research_agency_lab/experiments/learners"
LEARNER_ID = __import__("re").compile(r"[A-Za-z0-9_-]{1,64}")

PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qaari — recitation analysis</title>
<style>
 :root{--bg:#fbfaf7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0d8;--ok:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--card:#fff}
 @media(prefers-color-scheme:dark){:root{--bg:#14140f;--fg:#ececec;--mut:#9a9a94;--line:#2e2e28;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--card:#1c1c17}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:900px;margin:0 auto;padding:32px 16px 80px}
 h1{font-size:1.5rem;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 28px;font-size:.92rem}
 form{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:24px}
 label{display:block;font-size:.82rem;color:var(--mut);margin:14px 0 5px;text-transform:uppercase;letter-spacing:.04em}
 input,select{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit}
 .row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:120px}
 button{margin-top:18px;width:100%;padding:13px;border:0;border-radius:8px;background:var(--fg);color:var(--bg);font:600 1rem inherit;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
 .note{font-size:.84rem;color:var(--mut);margin-top:10px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
 .k{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line);font-size:.92rem}
 .k:last-child{border:0} .k span:last-child{font-variant-numeric:tabular-nums}
 .err{color:var(--bad)} .ok{color:var(--ok)} .warn{color:var(--warn)}
 .ar{font-size:1.3rem;direction:rtl}
 table{width:100%;border-collapse:collapse;font-size:.88rem}
 th{text-align:left;color:var(--mut);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--line)}
 td{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
 code{font:.86em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);padding:1px 5px;border-radius:4px}
 #status{margin-top:14px;font-size:.9rem;color:var(--mut)}
 details summary{cursor:pointer;color:var(--mut);font-size:.9rem;padding:4px 0}
 .tabs{display:flex;gap:4px;margin-bottom:20px;border-bottom:1px solid var(--line)}
 .tabs button{width:auto;margin:0;padding:9px 16px;background:transparent;color:var(--mut);border:0;border-bottom:2px solid transparent;font-weight:500;font-size:.95rem}
 .tabs button.on{color:var(--fg);border-bottom-color:var(--fg)}
 .pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:.76rem;font-weight:600}
 .p-covered,.p-ship{background:#1a7f4b22;color:var(--ok)}
 .p-partial,.p-caution{background:#8a610022;color:var(--warn)}
 .p-uncovered,.p-absent,.p-broken{background:#b3261e22;color:var(--bad)}
 .p-blocked,.p-hold,.p-out_of_scope,.p-suspect,.p-plausible,.p-unmeasured{background:#6b6b6b22;color:var(--mut)}
 .bar{height:7px;border-radius:4px;background:var(--line);overflow:hidden;margin:10px 0 18px}
 .bar>i{display:block;height:100%;background:var(--ok)}
</style>
<div class="wrap">
<h1>Qaari — recitation analysis</h1>
<p class="sub">Upload a recitation and every letter is scored: identity, its five to seven classical
sifāt, timing in your own counts, and every located tajweed rule graded against what it requires.
<a href="/protocol">Recording protocol →</a> · <a href="/review">Listening review →</a></p>

<div class="tabs">
  <button class="on" data-t="analyse">Analyse a recitation</button>
  <button data-t="capability">What the engine can judge</button>
</div>
<div id="t-analyse">
<form id="f">
  <label for="audio">Recitation (mp3 / wav / m4a)</label>
  <input id="audio" name="audio" type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg" required>
  <div class="row">
    <div><label for="surah">Surah</label><input id="surah" name="surah" type="number" min="1" max="114" placeholder="e.g. 23"></div>
    <div><label for="ayah">From ayah</label><input id="ayah" name="ayah" type="number" min="1" placeholder="41"></div>
    <div><label for="ayah_end">To ayah</label><input id="ayah_end" name="ayah_end" type="number" min="1" placeholder="optional"></div>
  </div>
  <label for="learner">Learner id (optional — keeps your history so every recitation sharpens the picture)</label>
  <input id="learner" name="learner" type="text" maxlength="64" placeholder="e.g. abdul">
  <button id="go" type="submit">Analyse</button>
  <button id="det" type="button" style="margin-top:8px;background:transparent;color:var(--fg);border:1px solid var(--line)">Detect verses from the audio</button>
  <div id="cands"></div>
  <div id="status"></div>
  <p class="note">Inference runs on this box's CPU. Detection transcribes the first 25 s with
  whisper-tiny-ar-quran and offers the matches — it does not choose for you, because top-1 is 0.88
  while top-3 is 1.00 on corpus clips.</p>
</form>
<div id="out"></div>
</div>
<div id="t-capability" style="display:none"><div class="card">Loading…</div></div>
</div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out'),st=document.getElementById('status'),go=document.getElementById('go');
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
f.onsubmit=async e=>{
 e.preventDefault(); out.innerHTML=''; go.disabled=true;
 const t0=Date.now(); let n=0;
 const tick=setInterval(()=>{n=((Date.now()-t0)/1000).toFixed(0);st.textContent='Analysing… '+n+'s (model is warm; expect ~4 s per verse)';},500);
 try{
   const r=await fetch('/analyze',{method:'POST',body:new FormData(f)});
   const j=await r.json();
   clearInterval(tick); st.textContent='Done in '+((Date.now()-t0)/1000).toFixed(1)+'s';
   if(!r.ok||j.error){out.innerHTML='<div class="card err">'+esc(j.error||('HTTP '+r.status))+(j.detail?'<pre>'+esc(j.detail)+'</pre>':'')+'</div>';return;}
   render(j);
 }catch(err){clearInterval(tick);st.textContent='';out.innerHTML='<div class="card err">'+esc(err)+'</div>';}
 finally{go.disabled=false;}
};
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
 document.getElementById('t-analyse').style.display=b.dataset.t==='analyse'?'':'none';
 document.getElementById('t-capability').style.display=b.dataset.t==='capability'?'':'none';
 if(b.dataset.t==='capability')loadCap();
});
let capLoaded=false;
async function loadCap(){
 if(capLoaded)return; capLoaded=true;
 const el=document.getElementById('t-capability');
 try{
  const j=await (await fetch('/capability')).json();
  const s=j.summary||{};
  let h='<div class="card"><h3 style="margin:0 0 4px">Treatise coverage</h3>';
  h+='<div class="note">'+s.concepts+' concepts from the treatises · '+s.covered_pct+'% fully covered</div>';
  h+='<div class="bar"><i style="width:'+s.covered_pct+'%"></i></div>';
  for(const[k,v]of Object.entries(s.by_status||{}))h+=kv(k,v);
  h+=kv('Rules catalogued',s.rules_catalogued);
  h+=kv('Rule instances located',(s.rule_instances_located||0).toLocaleString());
  h+='</div>';

  if((j.matrix||[]).length){
   h+='<div class="card"><h3 style="margin:0 0 10px">Capability — measured against target</h3><table>';
   h+='<tr><th>Capability</th><th>Detects</th><th>Now</th><th>Target</th><th>Status</th></tr>';
   for(const m of j.matrix){
    h+='<tr><td><code>'+esc(m.capability)+'</code></td><td>'+esc(m.detects||'')+'</td>'
      +'<td>'+(m.value==null?'—':(+m.value).toFixed(4))+'</td>'
      +'<td>'+(m.target==null?'—':(+m.target).toFixed(3))+'</td>'
      +'<td><span class="pill p-'+esc(m.readiness)+'">'+esc(m.readiness)+'</span></td></tr>';
    if(m.gap&&m.gap!=='—')h+='<tr><td colspan="5" class="note" style="padding-top:0">'+esc(m.gap)+'</td></tr>';
   }
   h+='</table></div>';
  }

  h+='<div class="card"><h3 style="margin:0 0 10px">Every tajweed rule the parser locates</h3><table>';
  h+='<tr><th>Rule</th><th>Family</th><th>Judged by</th><th>Instances</th><th>Status</th></tr>';
  for(const r of j.rules){
   h+='<tr><td><code>'+esc(r.rule_type)+'</code></td><td>'+esc(r.family)+'</td>'
     +'<td>'+esc(r.mechanism)+(r.muaalem_head?' · '+esc(r.muaalem_head):'')+'</td>'
     +'<td>'+(r.n_instances||0).toLocaleString()+'</td>'
     +'<td><span class="pill p-'+esc(r.status)+'">'+esc(r.status)+'</span></td></tr>';
   h+='<tr><td colspan="5" class="note" style="padding-top:0">'+esc(r.ph_signature||'')+' → test: '+esc(r.counterfactual||'')+'</td></tr>';
  }
  h+='</table></div>';

  h+='<div class="card"><h3 style="margin:0 0 10px">Treatise concepts</h3><table>';
  h+='<tr><th>Concept</th><th>Treatise</th><th>Mechanism</th><th>Status</th></tr>';
  for(const t of j.taxonomy){
   h+='<tr><td><b>'+esc(t.name)+'</b><br><code>'+esc(t.concept_id)+'</code></td>'
     +'<td>'+esc(t.treatise)+' §'+esc(t.section)+'</td><td>'+esc(t.mechanism)+'</td>'
     +'<td><span class="pill p-'+esc(t.status)+'">'+esc(t.status)+'</span></td></tr>';
   if(t.note)h+='<tr><td colspan="4" class="note" style="padding-top:0">'+esc(t.note)+'</td></tr>';
  }
  h+='</table></div>';
  el.innerHTML=h;
 }catch(e){el.innerHTML='<div class="card err">'+esc(e)+'</div>';capLoaded=false;}
}

const det=document.getElementById('det'),cands=document.getElementById('cands');
det.onclick=async()=>{
 const fi=document.getElementById('audio');
 if(!fi.files.length){st.textContent='Choose a file first.';return;}
 det.disabled=true;cands.innerHTML='';
 const t0=Date.now();
 const tick=setInterval(()=>{st.textContent='Transcribing… '+((Date.now()-t0)/1000).toFixed(0)+'s';},500);
 try{
  const fd=new FormData();fd.append('audio',fi.files[0]);
  const r=await fetch('/detect',{method:'POST',body:fd});const j=await r.json();
  clearInterval(tick);
  if(!r.ok||j.error){st.textContent='';cands.innerHTML='<div class="card err">'+esc(j.error||'failed')+'</div>';return;}
  st.textContent='Transcribed in '+j.elapsed_seconds+'s';
  let h='<div class="card"><div class="note" style="margin:0 0 8px">Heard: <span class="ar">'+esc(j.transcript)+'</span></div>';
  h+='<div class="note" style="margin:0 0 8px">Pick the verse this starts at:</div>';
  for(const c of j.candidates)
   h+='<div class="k"><span><button type="button" class="pick" data-s="'+c.surah+'" data-a="'+c.ayah+'" style="width:auto;margin:0;padding:5px 12px;font-size:.85rem">'+c.surah+':'+c.ayah+'</button></span><span>match '+(100*c.score).toFixed(0)+'%</span></div>';
  h+='</div>';cands.innerHTML=h;
  document.querySelectorAll('.pick').forEach(b=>b.onclick=()=>{
   document.getElementById('surah').value=b.dataset.s;
   document.getElementById('ayah').value=b.dataset.a;
   st.textContent='Set to '+b.dataset.s+':'+b.dataset.a+' — press Analyse.';
  });
 }catch(e){clearInterval(tick);st.textContent='';cands.innerHTML='<div class="card err">'+esc(e)+'</div>';}
 finally{det.disabled=false;}
};
function kv(label,val,cls){return '<div class="k"><span>'+esc(label)+'</span><span class="'+(cls||'')+'">'+esc(val)+'</span></div>';}
function measures(m){
 if(!m)return '';
 const s=m.summary,r=m.recording||{};
 let h='<div class="card"><h3 style="margin:0 0 10px">Measurements</h3>';
 h+=kv('Words with every judgement passing',s.words_all_correct+' / '+s.words,s.words_all_correct==s.words?'ok':'warn');
 h+=kv('Characteristics not realised',s.characteristics_not_realised+' / '+s.characteristics_scored,s.characteristics_not_realised?'warn':'ok');
 h+=kv('Letters whose identity is not confirmed',s.identity_not_confirmed+' / '+s.letters,s.identity_not_confirmed?'warn':'ok');
 h+=kv('Rules failed',s.rules_failed+' / '+s.rules_scored,s.rules_failed?'warn':'ok');
 if(r.seconds_per_count)h+=kv('Tempo',r.seconds_per_count+' s per count ('+r.tempo_class+')');
 const L={},R={};m.letters.forEach(l=>L[l.id]=l);m.rules.forEach(x=>R[x.id]=x);
 h+='<table><tr><th>Word</th><th>Measurement</th><th>Expected</th><th>Observed</th><th>Margin / deviation</th><th>Masters pct</th><th>z</th></tr>';
 for(const w of m.words){ if(w.all_correct)continue;
  for(const f of w.failing){ let row;
   if(f.includes(':R')){const x=R[f];row=['rule '+x.rule,(x.expected_counts||[]).join('–')||'—',x.observed_counts??x.status,x.deviation_counts??'—',x.percentile_masters??'—',x.z_masters??'—'];}
   else{const i=f.lastIndexOf(':'),l=L[f.slice(0,i)],k=f.slice(i+1);
    if(k=='identity')row=[l.symbol+' identity',l.symbol,l.identity.competitor||'—',l.identity.margin,'—','—'];
    else{const c=l.characteristics[k];row=[l.symbol+' '+k,c.expected,c.observed,c.margin,c.percentile_masters??'—',c.z_masters??'—'];}}
   h+='<tr><td class="ar">'+esc(w.text)+'</td>'+row.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>';}}
 h+='</table></div>';
 return h;
}
function knowledge(k){
 if(!k)return '';
 const s=k.summary||{},sk=k.skills||{},pct=x=>(100*x).toFixed(1)+'%';
 let h='<div class="card"><h3 style="margin:0 0 4px">Skill estimates (learner model)</h3>'
  +'<p class="note" style="margin:0 0 10px">'+s.skills_measured+' skills measured, '+s.skills_inferred+' inferred from related evidence, '
  +s.skills_prior_only+' not yet seen'+(k.recitations_so_far>1?' — '+k.recitations_so_far+' recitations so far':'')+'.</p>';
 for(const[r,d]of Object.entries(sk).filter(([,d])=>d.basis=='measured').sort((a,b)=>b[1].gap_to_masters-a[1].gap_to_masters).slice(0,10))
  h+=kv(r+' ('+d.basis+')',pct(d.estimate)+' ['+pct(d.interval90[0])+'–'+pct(d.interval90[1])+'] · masters '+pct(d.masters??d.cohort)+' · gap '+(100*d.gap_to_masters).toFixed(1)+' pts',d.interval_below_masters?'warn':'');
 const p=k.path||{}; if((p.ready_to_learn||[]).length)h+=kv('Ready to learn next',p.ready_to_learn.slice(0,6).join(' → '));
 for(const[r,v]of Object.entries(k.lengths||{}))
  h+=kv('Your '+r.replace(/_/g,' '),v.median_counts+' counts'+(v.spread_counts!=null?' (± '+v.spread_counts+')':'')+(v.masters_median_counts!=null?'; masters '+v.masters_median_counts:''));
 const hr=(k.projection||{}).hardest_ayahs||[];
 if(hr.length)h+=kv('Where the Quran will test you most',hr.slice(0,5).map(a=>a.surah+':'+a.ayah+' ('+a.because.join(', ')+')').join(' · '));
 h+='<p class="note">'+esc(k.honesty)+'</p></div>';
 return h;
}
function render(j){
 const s=j.summary||{},m=j.mastery||{};
 let h=measures(j.measurements)+knowledge(j.knowledge)+'<div class="card"><h3 style="margin:0 0 10px">Summary</h3>';
 h+=kv('Ayahs',s.ayahs)+kv('Letters',s.letters)+kv('Judgments',s.judgments);
 h+=kv('Rules located',s.rules_located)+kv('Errors',s.errors,s.errors?'err':'ok');
 if(s.accuracy!=null)h+=kv('Accuracy',(100*s.accuracy).toFixed(1)+'%',s.accuracy>0.9?'ok':'warn');
 if(j.detected)h+=kv('Verses','auto-detected: '+esc(j.detected),'warn');
 h+='</div>';
 h+='<div class="card"><h3 style="margin:0 0 10px">Mastery</h3>';
 if(m.tempo_haraka_s)h+=kv('Tempo',m.tempo_mode+' ('+m.tempo_haraka_s+' s per count)');
 for(const[k,v]of Object.entries(m.taswiyah||{}))h+=kv('Taswiyah '+k,'median '+v.median_counts+' counts, CV '+v.cv);
 for(const[k,v]of Object.entries(j.ghunnah_grades||{}))h+=kv('Ghunnah '+k,v.n+' × median '+v.median_counts+' counts, '+(100*v.accuracy).toFixed(0)+'% in band');
 const ls=j.letter_strength||{}; if(ls.ratio!=null)h+=kv('Letter strength (quwwa)',ls.realised+'/'+ls.strong_sifat_expected+' strong sifāt ('+(100*ls.ratio).toFixed(1)+'%)');
 h+='</div>';
 if((j.errors||[]).length){
  h+='<div class="card"><h3 style="margin:0 0 10px">What to fix</h3><table><tr><th>Rule</th><th>Word</th><th>Finding</th></tr>';
  for(const e of j.errors.slice(0,40)){
   let d=e.status; const ev=e.evidence||{};
   if(ev.given_counts!=null)d='gave '+ev.given_counts+' counts, needs '+(ev.expected||[]).join('–');
   else if(ev.heard)d='heard '+ev.heard+' instead';
   h+='<tr><td><code>'+esc(e.rule)+'</code></td><td class="ar">'+esc(e.word)+'</td><td class="err">'+esc(d)+'</td></tr>';
  }
  h+='</table></div>';
 }
 h+='<div class="card"><h3 style="margin:0 0 10px">By rule</h3><table><tr><th>Rule</th><th>Attempted</th><th>Correct</th><th>Accuracy</th></tr>';
 for(const[k,v]of Object.entries(j.by_rule||{}))
  h+='<tr><td><code>'+esc(k)+'</code></td><td>'+v.attempted+'</td><td>'+v.pass+'</td><td class="'+(v.accuracy>0.9?'ok':'warn')+'">'+(v.accuracy==null?'—':(100*v.accuracy).toFixed(0)+'%')+'</td></tr>';
 h+='</table></div>';
 for(const a of (j.ayahs||[])){
  h+='<details class="card"><summary>'+a.surah+':'+a.ayah+' — '+a.letters.length+' letters, '+a.rules.length+' rules</summary>';
  h+='<table><tr><th>#</th><th>Letter</th><th>Counts</th><th>Identity</th><th>Sifāt not realised</th></tr>';
  for(const l of a.letters){
   const bad=Object.entries(l.sifat||{}).filter(([,v])=>!v.realised).map(([k,v])=>k+'→'+v.model_best).join(', ');
   h+='<tr><td>'+l.i+'</td><td class="ar">'+esc(l.symbol)+'</td><td>'+(l.duration_counts??'—')+'</td>'
     +'<td class="'+(l.identity.confirmed?'ok':'err')+'">'+(l.identity.confirmed?'ok':'heard '+esc(l.identity.heard_instead))+'</td>'
     +'<td class="err">'+esc(bad)+'</td></tr>';
  }
  h+='</table></details>';
 }
 out.innerHTML=h;
}
</script>
"""


class AudioDecodeError(ValueError):
    """The upload is not audio any decoder here can read -- the user's file, not a server fault."""


def _ffmpeg_decode(data: bytes):  # type: ignore[no-untyped-def]
    """Any container ffmpeg understands to 16 kHz mono float32.

    Input goes through a temp file, not a pipe: MP4/M4A (what phones record) usually keeps its index
    at the END of the file, and ffmpeg cannot seek back to it on a pipe.
    """
    import subprocess
    import tempfile

    import numpy as np
    with tempfile.NamedTemporaryFile(suffix=".upload") as f:
        f.write(data)
        f.flush()
        try:
            run = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", f.name,
                                  "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                                 capture_output=True, timeout=120, check=False)
        except FileNotFoundError as exc:
            raise AudioDecodeError("this audio format needs ffmpeg, which is not installed on the "
                                   "server") from exc
    wave = np.frombuffer(run.stdout, dtype="<f4")
    if run.returncode != 0 or wave.size == 0:
        why = run.stderr.decode(errors="replace").strip().splitlines()
        # ffmpeg names the temp file in its message; the user only needs the reason
        reason = why[-1].split(": ", 1)[-1] if why else ""
        raise AudioDecodeError("could not decode the uploaded audio"
                               + (f" ({reason})" if reason else "") + ". Try WAV, MP3, M4A, OGG or FLAC.")
    return wave.copy()


def decode_upload(data: bytes):  # type: ignore[no-untyped-def]
    """Any uploaded container to 16 kHz mono float32."""
    import io as _io
    import librosa
    import numpy as np
    import soundfile as sf
    try:
        wave, sr = sf.read(_io.BytesIO(data), dtype="float32", always_2d=False)
        if getattr(wave, "ndim", 1) > 1:
            wave = wave.mean(axis=1)
        if sr != 16000:
            wave = librosa.resample(np.asarray(wave, dtype="float32"), orig_sr=sr, target_sr=16000)
    except Exception:  # noqa: BLE001
        # libsndfile rejects AAC/M4A (what most phone recorders write, often named .wav), Opus in
        # WebM and odd WAV codecs. librosa.load on bytes only retries libsndfile, so it cannot
        # rescue these; ffmpeg reads them all.
        wave = _ffmpeg_decode(data)
    return np.asarray(wave, dtype="float32")


def create_app():  # type: ignore[no-untyped-def]
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    api = FastAPI(title="qaari-upload", version="1.0.0")
    state: dict[str, Any] = {"engine": None, "warm": False, "verse_id_ok": False}

    def engine():  # type: ignore[no-untyped-def]
        if state["engine"] is None:
            from app.engine import Engine
            eng = Engine()
            eng.posteriors(__import__("numpy").zeros(16000, dtype="float32"))  # force model + layout
            state["engine"] = eng
            state["warm"] = True
        return state["engine"]

    @api.on_event("startup")
    def warm() -> None:
        # 2.4 GB of weights: load now, not on the user's first upload
        try:
            engine()
        except Exception:  # noqa: BLE001 - serve the page even if warming fails, and say so
            traceback.print_exc()

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_warm": state["warm"],
                "detect_endpoint": "/detect", "port": PORT}

    @api.get("/capability")
    def capability():  # type: ignore[no-untyped-def]
        """What the engine can actually judge right now, read live from the store.

        Three tables generated from one source each, so this cannot drift from what was measured:
        `tajweed_taxonomy` (the treatises' concepts and whether each is covered), `rule_catalogue`
        (the parser's rules and the mechanism that judges each), and `launch_matrix` (per-capability
        readiness with today's number against its target).
        """
        from datastore.store import connect
        con = connect(read_only=True)

        def rows(sql):  # type: ignore[no-untyped-def]
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

        tax = rows("SELECT concept_id, treatise, section, name, mechanism, implementation, status, "
                   "note FROM tajweed_taxonomy ORDER BY status, treatise, section")
        rules = rows("SELECT rule_type, family, mechanism, ph_signature, counterfactual, "
                     "muaalem_head, n_instances, legacy_pass_rate, status FROM rule_catalogue "
                     "ORDER BY family, rule_type")
        try:
            matrix = rows("SELECT capability, family, level, detects, metric, value, target, n, "
                          "readiness, gap FROM launch_matrix ORDER BY readiness, family, capability")
        except Exception:  # noqa: BLE001 - the matrix is regenerated, may be absent
            matrix = []
        counts = {}
        for r in tax:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        scoped = sum(v for k, v in counts.items() if k != "out_of_scope")
        return JSONResponse({
            "taxonomy": tax, "rules": rules, "matrix": matrix,
            "summary": {
                "concepts": len(tax), "by_status": counts,
                "covered_pct": round(100 * counts.get("covered", 0) / scoped, 1) if scoped else None,
                "rules_catalogued": len(rules),
                "rule_instances_located": sum(r["n_instances"] or 0 for r in rules),
            },
        })

    @api.post("/detect")
    async def detect_verses(audio: UploadFile = File(...)):  # type: ignore[no-untyped-def]  # noqa: B008
        """Transcribe the head of a recording and offer the verses it might be.

        Deliberately does NOT pick one. Measured on corpus clips where the truth is known, top-1 is
        0.88 but top-3 is 1.00 — so the honest interface is a short list to choose from, not a silent
        guess that is wrong one time in eight.
        """
        data = await audio.read()
        if not data:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        try:
            wave = decode_upload(data)
            from app.verse_detect import detect as detect_fn
            t0 = time.time()
            cands = detect_fn(wave)
            return JSONResponse({
                "transcript": cands[0].text if cands else "",
                "candidates": [c.to_dict() for c in cands[:5]],
                "audio_seconds": round(float(wave.size) / 16000, 2),
                "elapsed_seconds": round(time.time() - t0, 2),
            })
        except AudioDecodeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=415)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}",
                                 "detail": traceback.format_exc()[-1200:]}, status_code=500)

    # -- the recording protocol ------------------------------------------------------------------
    def _words_of(surah: int, ayah: int) -> list[str]:
        Aya = engine()._phonetizer[0]
        return Aya(surah, ayah).get().uthmani.split()

    def _resolved() -> dict[str, list[dict[str, Any]]]:
        if "protocol" not in state:
            from app.protocol import resolve
            state["protocol"] = resolve(_words_of)
        return state["protocol"]

    @api.get("/protocol", response_class=HTMLResponse)
    def protocol_page() -> str:
        from app.protocol_page import PROTOCOL_PAGE
        return PROTOCOL_PAGE

    @api.get("/protocol/tests")
    def protocol_tests() -> dict[str, Any]:
        from app.protocol import TESTS
        res = _resolved()
        return {"tests": [{"id": t.id, "title": t.title, "surah": t.surah, "focus": t.focus,
                           "correct_note": t.correct_note, "b_is_correct": t.b_is_correct,
                           "b_note": t.b_note, "mistakes": res[t.id],
                           "verses": [{"ayah": a, "words": _words_of(t.surah, a)}
                                      for a in range(t.ayahs[0], t.ayahs[1] + 1)]}
                          for t in TESTS]}

    @api.get("/protocol/status")
    def protocol_status() -> dict[str, Any]:
        import json as _json
        out: dict[str, Any] = {}
        for d in sorted(PROTOCOL_DIR.glob("t*/*")) if PROTOCOL_DIR.is_dir() else []:
            cards = sorted(d.glob("*.scorecard.json"))
            if cards:
                out.setdefault(d.parent.name, {})[d.name] = {
                    "n": len(cards), "last": _json.loads(cards[-1].read_text())}
        return out

    @api.post("/protocol/submit")
    async def protocol_submit(
        audio: UploadFile = File(...),  # noqa: B008
        test_id: str = Form(...),  # noqa: B008
        take: str = Form(...),  # noqa: B008
    ):  # type: ignore[no-untyped-def]
        import json as _json
        from app.protocol import TESTS, scorecard
        test = next((t for t in TESTS if t.id == test_id), None)
        if test is None or take not in ("correct", "mistakes"):
            return JSONResponse({"error": "unknown test or take"}, status_code=422)
        data = await audio.read()
        if not data:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "Audio file too large (50 MB limit)"}, status_code=413)
        t0 = time.time()
        try:
            wave = decode_upload(data)
            verses = [(test.surah, a) for a in range(test.ayahs[0], test.ayahs[1] + 1)]
            report = engine().analyze(wave, verses)
            card = scorecard(test, take, report, _resolved()[test.id])
        except AudioDecodeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=415)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}",
                                 "detail": traceback.format_exc()[-1200:]}, status_code=500)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        d = PROTOCOL_DIR / test.id / take
        d.mkdir(parents=True, exist_ok=True)
        ext = Path(audio.filename or "take").suffix.lower() or ".bin"
        (d / f"{stamp}{ext}").write_bytes(data)
        (d / f"{stamp}.report.json").write_text(_json.dumps(report, ensure_ascii=False))
        (d / f"{stamp}.scorecard.json").write_text(_json.dumps(card, ensure_ascii=False))
        return JSONResponse({"scorecard": card, "elapsed_seconds": round(time.time() - t0, 2)})

    # -- the listening review --------------------------------------------------------------------
    @api.get("/review", response_class=HTMLResponse)
    def review_page() -> str:
        from app.review_page import REVIEW_PAGE
        return REVIEW_PAGE

    @api.get("/review/next")
    def review_next(limit: int = 20) -> dict[str, Any]:
        from app import review
        from datastore.reciter_profile import tempo_class
        items = []
        for c in review.queue(max(1, min(limit, 100))):
            try:
                c["verse_words"] = _words_of(c["surah"], c["ayah"])
            except Exception:  # noqa: BLE001 - the claim still stands without the verse text
                c["verse_words"] = []
            c["tempo_class"] = tempo_class(c.get("haraka_s"))
            items.append(c)
        return {"items": items}

    @api.post("/review/label")
    def review_label(id: str = Form(...), verdict: str = Form(...),  # noqa: A002, B008
                     note: str = Form("")):  # type: ignore[no-untyped-def]  # noqa: B008
        from app import review
        try:
            return review.add_label(id, verdict, note)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    @api.get("/review/stats")
    def review_stats() -> dict[str, Any]:
        from app import review
        return review.stats()

    @api.get("/review/audio/{cid}")
    def review_audio(cid: str, whole: int = 0):  # type: ignore[no-untyped-def]
        import subprocess
        from fastapi.responses import Response
        from app import review
        from datastore.review_queue import audio_path
        c = next((x for x in review._read(review.CANDIDATES) if x["id"] == cid), None)
        if c is None:
            return JSONResponse({"error": "unknown candidate"}, status_code=404)
        src = audio_path(c["speaker"], c["surah"], c["ayah"])
        if not src.is_file():
            return JSONResponse({"error": "audio unavailable"}, status_code=404)
        if whole:
            return Response(src.read_bytes(), media_type="audio/mpeg")
        # the word with enough context either side to hear it in its place
        lo, hi = max(0.0, c["start_s"] - 0.8), c["end_s"] + 0.8
        run = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{lo:.3f}", "-i",
                              str(src), "-t", f"{hi - lo:.3f}", "-f", "mp3", "pipe:1"],
                             capture_output=True, timeout=30, check=False)
        return Response(run.stdout, media_type="audio/mpeg")

    @api.post("/analyze")
    async def analyze(
        audio: UploadFile = File(...),  # noqa: B008
        surah: int | None = Form(None),  # noqa: B008
        ayah: int | None = Form(None),  # noqa: B008
        ayah_end: int | None = Form(None),  # noqa: B008
        learner: str | None = Form(None),  # noqa: B008
    ):  # type: ignore[no-untyped-def]
        data = await audio.read()
        if not data:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "Audio file too large (50 MB limit)"}, status_code=413)
        if learner and not LEARNER_ID.fullmatch(learner):
            return JSONResponse({"error": "learner id: letters, digits, - and _ only (max 64)"}, status_code=422)
        if not surah or not ayah:
            return JSONResponse({"error": "Give a surah and a starting ayah, or press Detect verses "
                                          "to have the recording transcribed and choose from the "
                                          "matches."}, status_code=422)

        t0 = time.time()
        try:
            wave = decode_upload(data)
            if wave.size < 1600:
                return JSONResponse({"error": "Recording is shorter than 0.1 s"}, status_code=422)

            verses = [(int(surah), a) for a in range(int(ayah), int(ayah_end or ayah) + 1)]
            if len(verses) > 60:
                return JSONResponse({"error": f"{len(verses)} ayahs is beyond this box's CPU budget; "
                                              "try 60 or fewer."}, status_code=422)
            report = engine().analyze(wave, verses)
            report["audio_seconds"] = round(float(wave.size) / 16000, 2)
            # the knowledge report: this recitation plus the learner's history, projected on the Quran
            from app.knowledge import build
            import json as _json
            hist_path = LEARNERS_DIR / f"{learner}.json" if learner else None
            history = _json.loads(hist_path.read_text()) if hist_path and hist_path.is_file() else None
            knowledge = build(report, history)
            if hist_path:
                LEARNERS_DIR.mkdir(parents=True, exist_ok=True)
                hist_path.write_text(_json.dumps(knowledge.pop("history")))
            else:
                knowledge.pop("history")
            report["knowledge"] = knowledge
            report["elapsed_seconds"] = round(time.time() - t0, 2)
            return JSONResponse(report)
        except AudioDecodeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=415)
        except Exception as exc:  # noqa: BLE001 - the page must show why, not a blank 500
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}",
                                 "detail": traceback.format_exc()[-1200:]}, status_code=500)

    return api


def main() -> int:
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=PORT, log_level="info")  # noqa: S104
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
