"""The calibration-sessions page: a round of 2-3 exercises, each recited to a spec and with scripted
mistakes, scored number by number against what each recording should produce (app/sessions.py)."""

SESSIONS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calibration Sessions</title>
<style>
 :root{--bg:#fbfaf7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0d8;--ok:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--card:#fff;--mark:#fde68a55}
 @media(prefers-color-scheme:dark){:root{--bg:#14140f;--fg:#ececec;--mut:#9a9a94;--line:#2e2e28;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--card:#1c1c17;--mark:#fbbf2433}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:920px;margin:0 auto;padding:32px 16px 80px}
 h1{font-size:1.5rem;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 20px;font-size:.92rem}
 a{color:inherit}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:18px}
 .card h2{font-size:1.05rem;margin:0 0 6px}
 .goal{font-size:.9rem;margin:0 0 6px} .learn{color:var(--mut);font-size:.82rem;margin:0 0 10px}
 .ar{font-size:1.45rem;line-height:2.3;direction:rtl;text-align:right;margin:8px 0}
 .ar .n{color:var(--mut);font-size:.9rem}
 .ar .stop{font-size:.8rem;color:#b3261e;font-weight:600;white-space:nowrap}
 .ar mark{background:var(--mark);color:inherit;border-radius:4px;padding:0 3px}
 .ar sup{font-size:.7rem;color:var(--warn);font-family:ui-sans-serif,system-ui}
 .takes{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
 @media(max-width:680px){.takes{grid-template-columns:1fr}}
 .take{border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
 .take h3{margin:0 0 6px;font-size:.92rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
 ol,ul{margin:6px 0 0;padding-left:20px;font-size:.88rem} li{margin-bottom:4px}
 .mut{color:var(--mut);font-size:.8rem}
 button{padding:9px 12px;border:0;border-radius:8px;background:var(--fg);color:var(--bg);font:600 .9rem inherit;cursor:pointer;margin:8px 6px 0 0}
 button.rec{background:var(--bad);color:#fff}
 input[type=file]{font-size:.85rem;margin-top:8px;max-width:100%}
 .phone{display:inline-block;padding:9px 12px;border-radius:8px;background:var(--bad);color:#fff;font:600 .9rem inherit;cursor:pointer;margin:8px 6px 0 0}
 .phone input{display:none}
 .st{font-size:.85rem;color:var(--mut);margin-top:8px;min-height:1.2em}
 .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
 .tbl{overflow-x:auto} table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:6px}
 td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
 th{color:var(--mut);font-weight:500} td.arw{direction:rtl;font-size:1rem}
 .done{color:var(--ok);font-size:.8rem;font-weight:600}
 .hist-btn{position:fixed;top:12px;left:12px;z-index:30;background:var(--card);color:var(--fg);border:1px solid var(--line);margin:0}
 .panel{position:fixed;top:0;left:0;bottom:0;width:300px;max-width:85vw;background:var(--card);border-right:1px solid var(--line);
   transform:translateX(-105%);transition:transform .2s;z-index:40;overflow-y:auto;padding:16px}
 .panel.open{transform:none} .panel h3{margin:0 0 10px;font-size:1rem}
 .panel .rd{margin-bottom:14px} .panel .rd b{display:block;cursor:pointer;margin-bottom:4px}
 .panel .ex{font-size:.82rem;color:var(--mut);padding:2px 0 2px 8px;cursor:pointer}
 .panel .cur{color:var(--ok);font-size:.75rem}
 .shade{position:fixed;inset:0;background:#0005;z-index:35;display:none} .shade.open{display:block}
 .wrap{padding-top:56px}
 .lq{border-top:1px solid var(--line);padding:12px 0 4px} .lq:first-of-type{border-top:0}
 .lq .q{font-weight:600;font-size:.95rem} .lq .why{color:var(--mut);font-size:.8rem;margin:2px 0 8px}
 .clip{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;padding:6px 0;border-bottom:1px dashed var(--line)}
 .clip .who{min-width:150px;font-size:.85rem} .clip audio{height:34px;max-width:100%}
 .clip button{margin:0;padding:6px 10px;font-size:.82rem;background:var(--card);color:var(--fg);border:1px solid var(--line)}
 .clip button.on{background:var(--fg);color:var(--bg)}
 .cmp{color:var(--mut);font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;margin-left:6px}
</style></head><body><div class="wrap">
<button class="hist-btn" onclick="panel(true)">☰ History</button>
<div class="shade" id="shade" onclick="panel(false)"></div>
<div class="panel" id="panel"><h3>Sessions</h3><div id="hist"></div></div>
<h1>Calibration sessions</h1>
<p class="sub">Each exercise is recited twice at <b>tadwīr</b>. <b>Take A</b> to the spec — the engine should measure every
length and characteristic inside the stated range. <b>Take B</b>, same speed, perfect except the numbered mistakes. Every take
is scored against what it should produce: expectations met, mistakes caught with the right kind, and anything flagged that you
did not do. <a href="/">analyser</a> · <a href="/protocol">protocol</a></p>
<h2 id="rtitle" style="font-size:1.1rem;margin:0 0 12px"></h2>
<p id="micnote" class="learn" style="display:none;border:1px solid var(--line);border-radius:8px;padding:8px 10px">
In-page recording needs a secure (https) page, and this one is plain http. Use <b>● Record with phone</b>: it opens
your phone's voice recorder, and the recording is uploaded and scored when you finish. Or <b>Choose File</b> to upload one.</p>
<div id="listen"></div>
<div id="ex">Loading…</div>
</div>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let EX=[];
// browsers expose the microphone only to https pages (or localhost); over plain http the in-page
// recorder cannot exist, so the phone's own recorder is offered instead (file input with capture)
const MIC=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia&&window.MediaRecorder);
let CUR=null,LATEST=null;
function panel(o){document.getElementById('panel').classList.toggle('open',o);document.getElementById('shade').classList.toggle('open',o);}
async function load(n){
  const r=await fetch('/sessions/rounds').then(r=>r.json());
  LATEST=r.rounds[r.rounds.length-1]; CUR=n||LATEST;
  document.getElementById('hist').innerHTML=r.history.slice().reverse().map(h=>'<div class="rd"><b onclick="go('+h.round+')">Round '+h.round
    +(h.round==LATEST?' <span class="cur">live</span>':'')+'</b>'+h.exercises.map(e=>'<div class="ex" onclick="go('+h.round+',&quot;'+e.id+'&quot;)">'
    +esc(e.title)+' — A '+e.takes.A+' · B '+e.takes.B+'</div>').join('')+'</div>').join('');
  document.getElementById('rtitle').textContent='Round '+CUR+(CUR==LATEST?' (live)':' (history)');
  const d=await fetch('/sessions/round/'+CUR).then(r=>r.json());
  EX=d.exercises;
  document.getElementById('ex').innerHTML=EX.map((x,i)=>card(x,i,d.status[x.id]||{})).join('');
  document.getElementById('listen').innerHTML=listen(d.listen||[]);
}
// questions for the expert's ear: a master's recitation, answered yes / no / unsure (saved as expert labels)
function listen(L){
  if(!L.length)return '';
  const n=L.reduce((a,q)=>a+q.clips.length,0), done=L.reduce((a,q)=>a+q.clips.filter(c=>c.answer).length,0);
  return '<div class="card"><h2>Listen and confirm <span class="mut" id="ldone">'+done+'/'+n+' answered</span></h2>'
   +'<p class="goal">Questions only your ear can settle. Each clip is the whole ayah; the word in question is its last. '
   +'<b>Yes</b> = heavy, <b>No</b> = light. Your answers are saved as expert labels.</p>'
   +L.map(q=>'<div class="lq"><div class="q">'+esc(q.question)+' <span class="ar" style="font-size:1.1rem">'+esc(q.word)+'</span> <span class="mut">('+esc(q.ref)+')</span>'
     +(q.comparison?'<span class="cmp">comparison</span>':'')+'</div><div class="why">'+esc(q.why)+'</div>'
     +q.clips.map(c=>'<div class="clip" id="l-'+esc(c.id)+'"><span class="who">'+esc(c.reciter)+'</span><audio controls preload="none" src="'+esc(c.audio)+'"></audio>'
       +['yes','no','unsure'].map(v=>'<button class="'+(c.answer==v?'on':'')+'" onclick="answer(this,&quot;'+esc(c.id)+'&quot;,&quot;'+v+'&quot;)">'+(v=='yes'?'Yes, heavy':v=='no'?'No, light':'Unsure')+'</button>').join('')
       +'</div>').join('')+'</div>').join('')+'</div>';
}
async function answer(b,id,v){
  const fd=new FormData();fd.append('id',id);fd.append('verdict',v);fd.append('note','round '+CUR+' listening');
  const r=await fetch('/review/label',{method:'POST',body:fd});
  if(!r.ok){alert('Not saved: '+r.status);return;}
  const row=b.parentNode, first=!row.querySelector('button.on');
  row.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));
  if(first){const e=document.getElementById('ldone'),m=e.textContent.match(/(\\d+)\\/(\\d+)/);e.textContent=(+m[1]+1)+'/'+m[2]+' answered';}
}
async function go(n,id){panel(false);await load(n);if(id){const e=document.getElementById('c-'+id);if(e)e.scrollIntoView();}}
function verses(x){
  return x.verses.map(v=>'<div class="ar">'+v.words.map((w,i)=>{
      const m=x.mistakes.findIndex(k=>k.ayah==v.ayah&&k.word==i);
      const s=(x.stops||[]).some(k=>k.ayah==v.ayah&&k.word==i)?' <span class="stop">⏹ stop, then resume</span><br>':'';
      return (m>=0?'<mark>'+esc(w)+'<sup>'+(m+1)+'</sup></mark>':esc(w))+s;
    }).join(' ')+' <span class="n">('+x.surah+':'+v.ayah+')</span></div>').join('');
}
function card(x,i,st){
  const words=(a,w)=>{const v=x.verses.find(v=>v.ayah==a);return v?v.words[w]:'';};
  const A='<p style="font-size:.88rem">'+esc(x.spec)+'</p><p class="mut">Declared wajh: <b>'+esc(x.wajh)+'</b>. The engine should measure:</p><ul>'
    +x.expect.map(e=>'<li><span class="ar" style="font-size:1rem">'+esc(words(e.ayah,e.word))+'</span> '+esc(e.rule)+(e.counts?' · '+e.counts[0]+'–'+e.counts[1]+' counts':' · passes')+'</li>').join('')+'</ul>';
  const B=x.b_is_correct?'<p style="font-size:.88rem">'+esc(x.b_spec)+'</p>'
    :'<p style="font-size:.88rem">Same speed, perfect except:</p><ol>'+x.mistakes.map(m=>'<li>'+esc(m.do)+'</li>').join('')+'</ol>';
  return '<div class="card" id="c-'+x.id+'"><h2>'+(i+1)+'. '+esc(x.title)+'</h2><p class="goal"><b>Goal.</b> '+esc(x.goal)+'</p><p class="learn">'+esc(x.learn)+'</p>'
    +verses(x)+'<div class="takes">'+take(x,'A','Take A · to the spec',A,st.A)+take(x,'B',x.b_is_correct?'Take B · faster, perfect':'Take B · scripted mistakes',B,st.B)+'</div></div>';
}
function take(x,k,h,body,st){
  const id=x.id+'-'+k;
  return '<div class="take"><h3>'+h+(st?' <span class="done">✓ '+st.n+' saved</span>':'')+'</h3>'+body
   +(MIC?'<button class="rec" id="r-'+id+'" onclick="rec(\\''+x.id+'\\',\\''+k+'\\')">● Record</button>':'')
   +'<label class="phone"><input type="file" accept="audio/*" capture onchange="up(\\''+x.id+'\\',\\''+k+'\\',this.files[0])">● Record with phone</label>'
   +'<input type="file" accept="audio/*" onchange="up(\\''+x.id+'\\',\\''+k+'\\',this.files[0])">'
   +'<div class="st" id="s-'+id+'">'+(st&&st.last?summ(st.last):'')+'</div><div class="tbl" id="o-'+id+'">'+(st&&st.last?detail(st.last):'')+'</div></div>';
}
function pair(p){
  return '<details open><summary>Two speeds compared — unit '+p.unit_s[0]+' s → '+p.unit_s[1]+' s (take B '+(p.speed_ratio??'?')+'× faster)</summary><table><tr><th>rule</th><th>seconds A → B</th><th>equivalent counts A → B</th><th>verdict A / B</th></tr>'
   +p.stretchings.map(s=>'<tr><td>'+esc(s.rule)+' <span class="mut">'+s.ayah+':'+s.word+'</span></td><td>'+s.seconds.join(' → ')+'</td><td>'+s.equivalent_counts.join(' → ')+'</td><td>'+s.verdict.join(' / ')+'</td></tr>').join('')+'</table></details>';
}
function tempo(c){const t=c.tempo;return 'tempo '+(t.seconds_per_count??'?')+' s/count ('+esc(t.class)+') '+(t.ok?'<span class="ok">✓</span>':'<span class="warn">outside tadwīr</span>');}
function summ(c){
  const fa=c.false_alarms.length, fac='<span class="'+(fa?'warn':'ok')+'">'+fa+' false-alarm word'+(fa==1?'':'s')+'</span>';
  if(c.expectations)return '<b>'+c.met+'/'+c.expectations.length+'</b> expectations met · '+fac+' · '+tempo(c);
  return '<b>'+c.caught+'/'+c.mistakes.length+'</b> mistakes caught · '+fac+' · '+tempo(c);
}
function detail(c){
  let h='';
  if(c.expectations){
    h+='<table><tr><th>word</th><th>rule</th><th>expected</th><th>measured</th><th>z vs masters</th><th></th></tr>'+c.expectations.map(e=>'<tr><td class="arw">'+esc(e.text)+'</td><td>'+esc(e.rule)+'</td><td>'+(e.expected_counts?e.expected_counts.join('–'):'pass')+'</td><td>'+(e.measured??'—')+(e.status&&e.status!='pass'?' ('+esc(e.status)+')':'')+'</td><td>'+(e.z_masters??'')+'</td><td class="'+(e.verdict=='ok'?'ok':'bad')+'">'+esc(e.verdict)+'</td></tr>').join('')+'</table>';
  }else{
    h+='<table><tr><th>#</th><th>word</th><th>result</th><th>engine evidence</th></tr>'+c.mistakes.map((m,i)=>'<tr><td>'+(i+1)+'</td><td class="arw">'+esc(m.text)+'</td><td class="'+(m.verdict=='caught'?'ok':m.verdict=='missed'?'bad':'warn')+'">'+esc(m.verdict)+'</td><td>'+esc((m.evidence.length?m.evidence:m.engine_failing).join('; '))+'</td></tr>').join('')+'</table>';
  }
  if(c.tempo_pair)h+=pair(c.tempo_pair);
  if(c.letter_matrix)h+=matrix(c.letter_matrix);
  if(c.false_alarms.length)h+='<details><summary>'+c.false_alarms.length+' word(s) flagged that the script left alone</summary><table>'+c.false_alarms.map(a=>'<tr><td class="arw">'+esc(a.text)+'</td><td>'+esc(a.failing.join('; '))+'</td></tr>').join('')+'</table></details>';
  return h;
}
// only what the engine measures: makhraj (the identity test) and the characteristics it has a head for
const COLS=[['hams_jahr','hams / jahr'],['shiddah_rakhawah','shiddah / rakhawah'],['istila_istifal','isti’la / istifal'],
 ['itbaq_infitah','itbaq / infitah'],['safir','safir'],['qalqalah','qalqalah'],['tafashshi','tafashshi'],['istitalah','istitalah'],['ghunnah','ghunnah']];
const AR={'[همس]':'hams','[جهر]':'jahr','[شديد]':'shiddah','[رخو]':'rakhawah','[بين بين]':'tawassut','[مفخم]':'heavy','[مرقق]':'light',
 '[مطبق]':'itbaq','[منفتح]':'infitah','[صفير]':'safir','[لا صفير]':'no safir','[مقلقل]':'qalqalah','[لا قلقلة]':'no qalqalah','[مغن]':'ghunnah','[لا غنة]':'no ghunnah'};
function heard(x){return AR[x]||x||'';}
function mcell(v){
  if(!v||!v.measured||!v.scored)return '<td></td>';
  return v.realised?'<td class="ok">✓ '+esc(v.value)+'<br><span class="mut">'+v.margin+'</span></td>'
                   :'<td class="bad">✗ heard '+esc(heard(v.observed))+'<br><span class="mut">'+v.margin+'</span></td>';
}
function matrix(M){
  const S=M.summary, g=Object.keys(S), lab=k=>k.startsWith('held')?'held (saakin / shaddah / stop)':k;
  let h='<details><summary>Letter matrix — makhraj and characteristics ('+S.all.letters+' letters)</summary><div class="tbl"><table><tr><th>accuracy</th>'+g.map(k=>'<th>'+esc(lab(k))+'</th>').join('')+'</tr>'
   +'<tr><td>makhraj</td>'+g.map(k=>'<td>'+(S[k].makhraj_confirmed==null?'':Math.round(100*S[k].makhraj_confirmed)+'%')+'</td>').join('')+'</tr>';
  const names=COLS.map(c=>c[0]).filter(n=>g.some(k=>S[k].sifat[n]));
  h+=names.map(n=>'<tr><td>'+esc(COLS.find(c=>c[0]==n)[1])+'</td>'+g.map(k=>{const x=S[k].sifat[n];return '<td>'+(x?Math.round(100*x.realised)+'% <span class="mut">('+x.n+')</span>':'')+'</td>';}).join('')+'</tr>').join('')+'</table>';
  const cols=COLS.filter(c=>names.includes(c[0]));
  h+='<table><tr><th>letter</th><th>context</th><th>makhraj</th>'+cols.map(c=>'<th>'+c[1]+'</th>').join('')+'</tr>'+M.letters.map(r=>{
    const by={};r.sifat.forEach(s=>by[s.sifah]=s);
    const mk=r.makhraj, mc=mk.margin==null?'<td></td>':'<td class="'+(mk.confirmed?'ok':'bad')+'" title="'+esc(mk.region)+'">'+(mk.confirmed?'✓':'✗ heard '+esc(mk.competitor))+'<br><span class="mut">'+mk.margin+(mk.confirmed&&mk.competitor?' vs '+esc(mk.competitor):'')+'</span></td>';
    return '<tr><td class="arw">'+esc(r.letter)+'</td><td>'+esc(r.context)+'</td>'+mc+cols.map(c=>mcell(by[c[0]])).join('')+'</tr>';}).join('')+'</table></div>'
   +'<p class="mut">Numbers are the engine’s confidence margin (higher = clearer). Makhraj: the letter against the nearest letter it could be confused with.</p></details>';
  return h;
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
  const fd=new FormData();fd.append('audio',f);fd.append('exercise',t);fd.append('take',k);
  try{
    const r=await fetch('/sessions/submit',{method:'POST',body:fd});const j=await r.json();
    if(!r.ok){s.innerHTML='<span class="bad">'+esc(j.error||r.status)+'</span>';return;}
    s.innerHTML=summ(j.score)+' <span class="mut">('+j.elapsed_seconds+' s)</span>';o.innerHTML=detail(j.score);
  }catch(e){s.innerHTML='<span class="bad">'+esc(e.message)+'</span>';}
}
if(!MIC)document.getElementById('micnote').style.display='block';
load();
</script></body></html>
"""
