"""The recording-protocol page: ten paired tests, recorded or uploaded, scored against the script."""

PROTOCOL_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recording Protocol</title>
<style>
 :root{--bg:#fbfaf7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0d8;--ok:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--card:#fff;--mark:#fde68a55}
 @media(prefers-color-scheme:dark){:root{--bg:#14140f;--fg:#ececec;--mut:#9a9a94;--line:#2e2e28;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--card:#1c1c17;--mark:#fbbf2433}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:900px;margin:0 auto;padding:32px 16px 80px}
 h1{font-size:1.5rem;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 20px;font-size:.92rem}
 a{color:inherit}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:18px}
 .card h2{font-size:1.05rem;margin:0 0 2px} .focus{color:var(--mut);font-size:.86rem;margin:0 0 12px}
 .ar{font-size:1.45rem;line-height:2.3;direction:rtl;text-align:right;margin:8px 0}
 .ar .n{color:var(--mut);font-size:.9rem}
 .ar mark{background:var(--mark);color:inherit;border-radius:4px;padding:0 3px}
 .ar sup{font-size:.7rem;color:var(--warn);font-family:ui-sans-serif,system-ui}
 .takes{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
 @media(max-width:640px){.takes{grid-template-columns:1fr}}
 .take{border:1px solid var(--line);border-radius:10px;padding:12px}
 .take h3{margin:0 0 6px;font-size:.92rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
 ol{margin:6px 0 0;padding-left:20px;font-size:.9rem} li{margin-bottom:4px}
 .rule{color:var(--mut);font-size:.8rem}
 button{padding:9px 12px;border:0;border-radius:8px;background:var(--fg);color:var(--bg);font:600 .9rem inherit;cursor:pointer;margin:8px 6px 0 0}
 button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
 button.rec{background:var(--bad);color:#fff}
 button:disabled{opacity:.5;cursor:default}
 input[type=file]{font-size:.85rem;margin-top:8px;max-width:100%}
 .st{font-size:.85rem;color:var(--mut);margin-top:8px;min-height:1.2em}
 .res{margin-top:10px;font-size:.88rem}
 .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
 .pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:.76rem;font-weight:600;border:1px solid var(--line)}
 table{width:100%;border-collapse:collapse;font-size:.84rem;margin-top:6px}
 td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
 th{color:var(--mut);font-weight:500}
 .done{color:var(--ok);font-size:.8rem;font-weight:600}
</style></head><body><div class="wrap">
<h1>Recording protocol</h1>
<p class="sub">Ten passages, each recited twice. <b>Take A</b>: as perfectly as you can. <b>Take B</b>: the same passage,
perfect everywhere <i>except</i> the numbered mistakes — make each one clearly and only there. Each take is scored the moment it
is uploaded: what the engine caught, what it missed, and anything it flagged that you did not do. Test 10 is two correct takes at
two speeds. <a href="/">← back to the analyser</a></p>
<div id="tests">Loading…</div>
</div>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let TESTS=[];
async function load(){
  const [t,s]=await Promise.all([fetch('/protocol/tests').then(r=>r.json()),fetch('/protocol/status').then(r=>r.json())]);
  TESTS=t.tests;
  document.getElementById('tests').innerHTML=TESTS.map(x=>card(x,s[x.id]||{})).join('');
}
function verses(x,marks){
  return x.verses.map(v=>'<div class="ar">'+v.words.map((w,i)=>{
      const m=marks?x.mistakes.findIndex(k=>k.ayah==v.ayah&&k.index==i):-1;
      return m>=0?'<mark>'+esc(w)+'<sup>'+(m+1)+'</sup></mark>':esc(w);
    }).join(' ')+' <span class="n">('+x.surah+':'+v.ayah+')</span></div>').join('');
}
function card(x,st){
  const b=x.b_is_correct?'<p>'+esc(x.b_note)+'</p>'
    :'<p>Recite the passage perfectly, except:</p><ol>'+x.mistakes.map(m=>'<li><span class="ar" style="font-size:1.1rem">'+esc(m.word)+'</span> — '+esc(m.do)+'<div class="rule">breaks: '+esc(m.rule)+'</div></li>').join('')+'</ol>';
  return '<div class="card" id="c-'+x.id+'"><h2>'+(TESTS.indexOf(x)+1)+'. '+esc(x.title)+'</h2><p class="focus">'+esc(x.focus)+'</p>'
    +verses(x,!x.b_is_correct)
    +'<div class="takes">'+take(x,'correct','Take A · correct','<p>'+esc(x.correct_note)+'</p>',st.correct)
    +take(x,'mistakes','Take B · '+(x.b_is_correct?'correct, fast':'scripted mistakes'),b,st.mistakes)+'</div></div>';
}
function take(x,k,h,body,st){
  const id=x.id+'-'+k;
  return '<div class="take"><h3>'+h+(st?' <span class="done">✓ '+st.n+' saved</span>':'')+'</h3>'+body
   +'<button class="rec" id="r-'+id+'" onclick="rec(\\''+x.id+'\\',\\''+k+'\\')">● Record</button>'
   +'<input type="file" accept="audio/*" id="f-'+id+'" onchange="up(\\''+x.id+'\\',\\''+k+'\\',this.files[0])">'
   +'<div class="st" id="s-'+id+'">'+(st&&st.last?summ(st.last):'')+'</div><div class="res" id="o-'+id+'"></div></div>';
}
function summ(c){
  if(!c)return'';
  const w=c.word_accuracy!=null?' · words fully correct '+Math.round(100*c.word_accuracy)+'%':'';
  return c.scripted?('<b>'+c.caught+'/'+c.scripted+'</b> caught'+(c.near?', '+c.near+' near':'')+', <span class="'+(c.missed?'bad':'ok')+'">'+c.missed+' missed</span>, '+c.false_alarm_words+' false-alarm words'+w)
    :('<span class="'+(c.false_alarm_words?'warn':'ok')+'">'+c.false_alarm_words+' words flagged</span> on a correct take'+w);
}
let media=null,chunks=[],recKey=null;
async function rec(t,k){
  const id=t+'-'+k,btn=document.getElementById('r-'+id);
  if(media&&recKey===id){media.stop();return;}
  if(media){alert('Stop the other recording first.');return;}
  try{
    const s=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    media=new MediaRecorder(s);chunks=[];recKey=id;
    media.ondataavailable=e=>chunks.push(e.data);
    media.onstop=()=>{s.getTracks().forEach(t=>t.stop());const b=new Blob(chunks,{type:media.mimeType});media=null;recKey=null;
      btn.textContent='● Record';up(t,k,new File([b],'take.'+(b.type.includes('mp4')?'m4a':'webm'),{type:b.type}));};
    media.start();btn.textContent='■ Stop and submit';
  }catch(e){alert('Microphone unavailable: '+e.message+' — upload a file instead.');}
}
async function up(t,k,f){
  if(!f)return;
  const id=t+'-'+k,s=document.getElementById('s-'+id),o=document.getElementById('o-'+id);
  s.textContent='Analysing '+(f.size/1024|0)+' KB…';o.innerHTML='';
  const fd=new FormData();fd.append('audio',f);fd.append('test_id',t);fd.append('take',k);
  try{
    const r=await fetch('/protocol/submit',{method:'POST',body:fd});const j=await r.json();
    if(!r.ok){s.innerHTML='<span class="bad">'+esc(j.error||r.status)+'</span>';return;}
    s.innerHTML=summ(j.scorecard)+' <span class="rule">('+j.elapsed_seconds+' s)</span>';
    o.innerHTML=detail(j.scorecard);
  }catch(e){s.innerHTML='<span class="bad">'+esc(e.message)+'</span>';}
}
function fl(fs){return fs.map(f=>esc(f.rule||f.sifah||('letter '+f.letter+(f.heard?' → '+f.heard:''))+(f.status?' ('+f.status+')':''))).join(', ');}
function detail(c){
  let h='';
  if(c.mistakes.length){
    h+='<table><tr><th>#</th><th>word</th><th>result</th><th>engine found</th></tr>'+c.mistakes.map((m,i)=>'<tr><td>'+(i+1)+'</td><td class="ar" style="font-size:1rem;line-height:1.4">'+esc(m.word)+'</td><td class="'+(m.verdict=='caught'?'ok':m.verdict=='near'?'warn':'bad')+'">'+m.verdict+(m.verdict=='caught'&&!m.right_kind?' (other kind)':'')+'</td><td>'+fl(m.engine_found)+'</td></tr>').join('')+'</table>';
  }
  if(c.false_alarms.length){
    h+='<details'+(c.mistakes.length?'':' open')+'><summary>'+c.false_alarms.length+' flagged words you did not script</summary><table>'+c.false_alarms.map(a=>'<tr><td>'+a.ayah+':'+(a.word_index+1)+'</td><td>'+fl(a.faults)+'</td></tr>').join('')+'</table></details>';
  }
  return h;
}
load();
</script></body></html>
"""
