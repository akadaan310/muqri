"""The listening-review page: one predicted mistake at a time, confirmed or rejected by ear."""

REVIEW_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Listening Review</title>
<style>
 :root{--bg:#fbfaf7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0d8;--ok:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--card:#fff;--mark:#fde68a66;--now:#93c5fd88}
 @media(prefers-color-scheme:dark){:root{--bg:#14140f;--fg:#ececec;--mut:#9a9a94;--line:#2e2e28;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--card:#1c1c17;--mark:#fbbf2433;--now:#3b82f666}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:760px;margin:0 auto;padding:28px 16px 80px}
 h1{font-size:1.4rem;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 18px;font-size:.9rem}
 a{color:inherit}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
 .meta{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;color:var(--mut);font-size:.85rem}
 .ar{font-size:1.6rem;line-height:2.2;direction:rtl;text-align:right;margin:14px 0}
 .ar mark{background:var(--mark);color:inherit;border-radius:5px;padding:0 4px}
 .ar .f{color:var(--bad);text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:10px}
 .ar .now{background:var(--now);border-radius:4px}
 .legend{font-size:.8rem;color:var(--mut);margin-top:-6px}
 .legend b{color:var(--bad)} .legend i{font-style:normal;background:var(--now);border-radius:3px;padding:0 4px}
 .claim{font-size:1.08rem;margin:6px 0 4px}
 .sev{display:inline-block;padding:1px 9px;border-radius:99px;font-size:.78rem;font-weight:600;border:1px solid var(--line)}
 .sev.major{color:var(--bad)} .sev.moderate{color:var(--warn)}
 .bar{height:6px;background:var(--line);border-radius:3px;margin:8px 0 16px;overflow:hidden}
 .bar i{display:block;height:100%;background:var(--warn)}
 .play{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
 button{padding:11px 14px;border:0;border-radius:8px;background:var(--fg);color:var(--bg);font:600 .95rem inherit;cursor:pointer}
 button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
 .verdict{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
 .verdict button{width:100%}
 .yes{background:var(--ok);color:#fff} .no{background:var(--bad);color:#fff} .unsure{background:var(--mut);color:#fff}
 textarea{width:100%;margin-top:10px;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit;min-height:56px}
 .keys{color:var(--mut);font-size:.8rem;margin-top:8px}
 .stats{margin-top:22px;font-size:.86rem}
 table{width:100%;border-collapse:collapse} td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
 th{color:var(--mut);font-weight:500}
 .empty{color:var(--mut);text-align:center;padding:40px 0}
</style></head><body><div class="wrap">
<h1>Listening review</h1>
<p class="sub">The engine proposes a mistake in a professional recitation. Listen, then say whether it is there. Lengths are in the
reciter's own counts; a count either way is calibration and is never proposed. Your verdicts calibrate every detector and become
training data. <a href="/">analyser</a> · <a href="/protocol">recording protocol</a></p>
<div id="card" class="card"><div class="empty">Loading…</div></div>
<div class="stats" id="stats"></div>
</div>
<audio id="au" preload="auto"></audio>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let Q=[],cur=null;
async function more(){ const j=await (await fetch('/review/next?limit=20')).json(); Q=j.items; show(); stats(); }
function show(){
  cur=Q.shift();
  const el=document.getElementById('card');
  if(!cur){el.innerHTML='<div class="empty">Nothing waiting. New candidates appear as more recitations are mined.</div>';return;}
  const words=cur.uthmani?verse(cur):(cur.verse_words||[]).map((w,i)=>i===cur.word_index?'<mark>'+esc(w)+'</mark>':esc(w)).join(' ');
  const pct=Math.min(100,Math.round(100*cur.magnitude/(cur.kind==='length'||cur.kind==='consistency'?4:12)));
  el.innerHTML='<div class="meta"><span>'+esc(cur.speaker.replace(/_/g,' '))+(cur.tempo_class?' · '+cur.tempo_class:'')+'</span><span>'+cur.surah+':'+cur.ayah+' · '+esc(cur.kind)+'</span></div>'
   +'<div class="ar" id="verse">'+words+'</div>'
   +'<div class="legend"><b>red</b> = the letters this finding is about · <i>blue</i> = what is sounding now</div>'
   +'<div class="claim">'+esc(cur.claim)+'</div><span class="sev '+cur.severity+'">'+cur.severity+' · magnitude '+cur.magnitude+'</span>'
   +'<div class="bar"><i style="width:'+pct+'%"></i></div>'
   +'<div class="play"><button onclick="play(0)">▶ the word</button><button class="ghost" onclick="play(1)">▶ whole verse</button></div>'
   +'<div class="verdict"><button class="yes" onclick="label(\\'yes\\')">1 · I hear it</button><button class="no" onclick="label(\\'no\\')">2 · Not there</button><button class="unsure" onclick="label(\\'unsure\\')">3 · Unsure</button></div>'
   +'<textarea id="note" placeholder="Note (optional): what you actually hear, how severe, the correct reading…"></textarea>'
   +'<div class="keys">Keys: space = word, v = verse, 1 / 2 / 3 = verdict (while not typing a note)</div>';
  play(0);
}
// the verse as grapheme clusters (a letter with its marks), so colouring never splits a mark from its
// letter; each cluster remembers which Uthmani character indices it holds
const MARK=/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]/;
let CL=[],OFFSET=0,RAF=null;
function verse(c){
  CL=[];const f=new Set(c.focus||[]);let h='';
  for(let i=0;i<c.uthmani.length;i++){
    const ch=c.uthmani[i];
    if(CL.length&&MARK.test(ch)){CL[CL.length-1].idx.push(i);CL[CL.length-1].t+=ch;continue;}
    CL.push({t:ch,idx:[i]});
  }
  CL.forEach((k,j)=>{const on=k.idx.some(i=>f.has(i));h+=k.t===' '?' ':'<span id="k'+j+'"'+(on?' class="f"':'')+'>'+esc(k.t)+'</span>';});
  return h;
}
const CHAR2CL=()=>{const m={};CL.forEach((k,j)=>k.idx.forEach(i=>m[i]=j));return m;};
function follow(){
  const a=document.getElementById('au');if(!cur||!cur.timeline)return;
  const m=CHAR2CL(),t=a.currentTime+OFFSET,now=new Set();
  for(const [s,e,cs] of cur.timeline) if(t>=s&&t<e) cs.forEach(c=>{if(m[c]!=null)now.add(m[c]);});
  CL.forEach((k,j)=>{const el=document.getElementById('k'+j);if(el)el.classList.toggle('now',now.has(j));});
  if(!a.paused&&!a.ended)RAF=requestAnimationFrame(follow);else CL.forEach((k,j)=>{const el=document.getElementById('k'+j);if(el)el.classList.remove('now');});
}
function play(whole){
  if(!cur)return; const a=document.getElementById('au');
  OFFSET=whole?0:Math.max(0,cur.start_s-0.8);   // the word clip starts 0.8 s before the letter
  a.src='/review/audio/'+cur.id+(whole?'?whole=1':''); a.onplay=()=>{cancelAnimationFrame(RAF);follow();};
  a.play().catch(()=>{});
}
async function label(v){
  if(!cur)return;
  const fd=new FormData();fd.append('id',cur.id);fd.append('verdict',v);fd.append('note',document.getElementById('note').value);
  const r=await fetch('/review/label',{method:'POST',body:fd});
  if(!r.ok){alert('Could not save: '+r.status);return;}
  if(!Q.length)await more();else{show();stats();}
}
async function stats(){
  const s=await (await fetch('/review/stats')).json();
  const rows=Object.entries(s.detectors).sort((a,b)=>b[1].decided-a[1].decided);
  document.getElementById('stats').innerHTML='<b>'+s.labelled+'</b> reviewed of '+s.candidates+' proposed.'
   +(rows.length?'<table><tr><th>detector</th><th>heard</th><th>not there</th><th>unsure</th><th>precision so far</th></tr>'+rows.map(([d,c])=>'<tr><td>'+esc(d)+'</td><td>'+c.yes+'</td><td>'+c.no+'</td><td>'+c.unsure+'</td><td>'+Math.round(100*c.precision)+'%</td></tr>').join('')+'</table>':'');
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA')return;
  if(e.key===' '){e.preventDefault();play(0);} else if(e.key==='v')play(1);
  else if(e.key==='1')label('yes'); else if(e.key==='2')label('no'); else if(e.key==='3')label('unsure');
});
more();
</script></body></html>
"""
