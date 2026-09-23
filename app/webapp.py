"""Upload page: drop a recitation in, get the full mastery report.

This is the engine's evaluation surface. Everything measured so far comes from the EveryAyah corpus;
this is where a real recording, recorded by a real person, goes through the same path.

    .venv/bin/python -m app.webapp            # serves on 0.0.0.0:8088

The acoustic model is 2.4 GB and takes ~40 s to load, so it is warmed at startup rather than on the
first upload — otherwise the first request looks broken. `GET /health` reports whether it is warm.

Inference is local CPU (4 cores, no GPU): fp32 RTF ~0.52, so one verse is a few seconds and a page of
recitation is one to two minutes. The page says so rather than appearing to hang.
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
</style>
<div class="wrap">
<h1>Qaari — recitation analysis</h1>
<p class="sub">Upload a recitation and every letter is scored: identity, its five to seven classical
sifāt, timing in your own counts, and every located tajweed rule graded against what it requires.</p>

<form id="f">
  <label for="audio">Recitation (mp3 / wav / m4a)</label>
  <input id="audio" name="audio" type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg" required>
  <div class="row">
    <div><label for="surah">Surah</label><input id="surah" name="surah" type="number" min="1" max="114" placeholder="e.g. 23"></div>
    <div><label for="ayah">From ayah</label><input id="ayah" name="ayah" type="number" min="1" placeholder="41"></div>
    <div><label for="ayah_end">To ayah</label><input id="ayah_end" name="ayah_end" type="number" min="1" placeholder="optional"></div>
  </div>
  <button id="go" type="submit">Analyse</button>
  <div id="status"></div>
  <p class="note">Inference runs on this box's CPU: about 4 s per verse, one to two minutes for a full
  page. Leave surah blank only if auto-detect is enabled below.</p>
</form>
<div id="out"></div>
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
function kv(label,val,cls){return '<div class="k"><span>'+esc(label)+'</span><span class="'+(cls||'')+'">'+esc(val)+'</span></div>';}
function render(j){
 const s=j.summary||{},m=j.mastery||{};
 let h='<div class="card"><h3 style="margin:0 0 10px">Summary</h3>';
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
                "auto_detect": state["verse_id_ok"], "port": PORT}

    @api.post("/analyze")
    async def analyze(
        audio: UploadFile = File(...),  # noqa: B008
        surah: int | None = Form(None),  # noqa: B008
        ayah: int | None = Form(None),  # noqa: B008
        ayah_end: int | None = Form(None),  # noqa: B008
    ):  # type: ignore[no-untyped-def]
        data = await audio.read()
        if not data:
            return JSONResponse({"error": "Empty upload"}, status_code=400)
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "Audio file too large (50 MB limit)"}, status_code=413)
        if not surah or not ayah:
            return JSONResponse(
                {"error": "Give a surah and a starting ayah. Automatic verse detection is not "
                          "enabled yet — app/verse_id.py has never been measured, and it will only "
                          "be offered once it is."}, status_code=422)

        import soundfile as sf
        import librosa
        import numpy as np

        t0 = time.time()
        try:
            try:
                wave, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                if getattr(wave, "ndim", 1) > 1:
                    wave = wave.mean(axis=1)
                if sr != 16000:
                    wave = librosa.resample(np.asarray(wave, dtype="float32"), orig_sr=sr, target_sr=16000)
            except Exception:  # noqa: BLE001 - mp3/m4a go through librosa
                wave, _ = librosa.load(io.BytesIO(data), sr=16000, mono=True)
            wave = np.asarray(wave, dtype="float32")
            if wave.size < 1600:
                return JSONResponse({"error": "Recording is shorter than 0.1 s"}, status_code=422)

            verses = [(int(surah), a) for a in range(int(ayah), int(ayah_end or ayah) + 1)]
            if len(verses) > 60:
                return JSONResponse({"error": f"{len(verses)} ayahs is beyond this box's CPU budget; "
                                              "try 60 or fewer."}, status_code=422)
            report = engine().analyze(wave, verses)
            report["audio_seconds"] = round(float(wave.size) / 16000, 2)
            report["elapsed_seconds"] = round(time.time() - t0, 2)
            return JSONResponse(report)
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
