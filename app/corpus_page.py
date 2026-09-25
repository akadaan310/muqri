"""The letters-corpus page: every letter in every form and every rule in every form, five instances per
reciter, playable, with the engine's measurements and a ranking per cell; and the perturbation lab --
a master's letter altered one characteristic at a time and measured again
(research_agency_lab/experiments/letter_corpus/)."""

CORPUS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Letters Corpus</title>
<style>
 :root{--bg:#fbfaf7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0d8;--ok:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--card:#fff;--chip:#f1efe8}
 @media(prefers-color-scheme:dark){:root{--bg:#14140f;--fg:#ececec;--mut:#9a9a94;--line:#2e2e28;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--card:#1c1c17;--chip:#26261f}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:980px;margin:0 auto;padding:24px 16px 80px}
 h1{font-size:1.5rem;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 14px;font-size:.9rem}
 a{color:inherit}
 .tabs{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
 .tabs button,.pick button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:7px 11px;font:600 .9rem inherit;cursor:pointer}
 .tabs button.on,.pick button.on{background:var(--fg);color:var(--bg)}
 .pick{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 14px;direction:rtl}
 .pick button{font-size:1.25rem;min-width:44px;padding:4px 8px}
 .pick.rules{direction:ltr} .pick.rules button{font-size:.8rem}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
 .card h2{font-size:1.05rem;margin:0 0 8px} .card h2 .ar{font-size:1.5rem;margin-left:6px}
 .rank{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
 .chip{background:var(--chip);border-radius:999px;padding:3px 10px;font-size:.78rem}
 .chip b{font-weight:700}
 .row{display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:start;border-top:1px solid var(--line);padding:8px 0}
 @media(max-width:640px){.row{grid-template-columns:1fr}}
 .who{font-size:.85rem;font-weight:600} .who .tag{font-weight:500;color:var(--warn);font-size:.72rem;margin-left:4px}
 .clips{display:flex;flex-wrap:wrap;gap:6px}
 .clip{border:1px solid var(--line);border-radius:8px;padding:5px 7px;min-width:92px;cursor:pointer;font-size:.75rem;background:var(--bg)}
 .clip .w{direction:rtl;font-size:1.05rem;display:block;text-align:right}
 .clip.playing{outline:2px solid var(--fg)}
 .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)} .mut{color:var(--mut);font-size:.8rem}
 .det{font-size:.78rem;margin-top:6px;white-space:pre-wrap;color:var(--mut)}
 table{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:6px}
 td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top} th{color:var(--mut);font-weight:500}
 label.tg{font-size:.85rem;margin-left:8px}
</style></head><body><div class="wrap">
<h1>Letters Corpus</h1>
<p class="sub">Every letter in every form and every rule in every form: five instances per reciter from the same places in the Qur'an, with what the engine measured on each. Tap an instance to play it and see its numbers. <a href="/sessions">Calibration sessions →</a></p>
<div class="tabs" id="tabs"></div>
<div id="bar"></div>
<div id="body"><p class="mut">Loading…</p></div>
</div>
<script>
let C=null,P=null,TAB='letters',SEL=null,RAW=false,cur=null;
const FORM={fatha:'fatḥah',kasra:'kasrah',damma:'ḍammah',fatha_long:'fatḥah + madd',kasra_long:'kasrah + madd',damma_long:'ḍammah + madd',sakin:'sākin',shadda:'shaddah',stop:'at the stop'};
const V={fatha:'َ',kasra:'ِ',damma:'ُ',fatha_long:'َا',kasra_long:'ِي',damma_long:'ُو',sakin:'ْ',shadda:'ّ',stop:'ْ ⏹'};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const name=r=>r=='learner'?'You':r.replace(/_(\\d+kbps|Murattal).*$/,'').replace(/_/g,' ');
async function init(){
  C=await (await fetch('/letters/corpus.json')).json();
  try{const r=await fetch('/letters/perturb.json');if(r.ok)P=await r.json();}catch(e){}
  tabs();show();
}
function tabs(){
  const t=[['letters','Letters'],['rules','Rules'],['best','Who reads it best'],['perturb','Perturbation lab']];
  document.getElementById('tabs').innerHTML=t.map(([k,l])=>'<button class="'+(TAB==k?'on':'')+'" onclick="TAB=\\''+k+'\\';SEL=null;tabs();show()">'+l+'</button>').join('')
    +'<label class="tg"><input type="checkbox" '+(RAW?'checked':'')+' onchange="RAW=this.checked"> raw audio (uncleaned)</label>';
}
function show(){
  const bar=document.getElementById('bar'),body=document.getElementById('body');
  if(TAB=='perturb'){bar.innerHTML='';body.innerHTML=perturb();return;}
  if(TAB=='best'){bar.innerHTML='';body.innerHTML=best();return;}
  const cells=C.cells.filter(c=>c.kind==(TAB=='letters'?'letter':'rule'));
  const keys=[...new Set(cells.map(c=>c.what))];
  if(!SEL)SEL=keys[0];
  bar.innerHTML='<div class="pick'+(TAB=='rules'?' rules':'')+'">'+keys.map(k=>'<button class="'+(k==SEL?'on':'')+'" onclick="SEL=\\''+k+'\\';show()">'+esc(TAB=='rules'?k.replace(/_/g,' '):k)+'</button>').join('')+'</div>';
  body.innerHTML=cells.filter(c=>c.what==SEL).map(card).join('')||'<p class="mut">No instances.</p>';
}
function card(c){
  const title=c.kind=='letter'?'<span class="ar">'+esc(c.what+(V[c.form]||''))+'</span> '+esc(FORM[c.form]||c.form)
    :esc(c.what.replace(/_/g,' '))+(c.form?' <span class="mut">'+esc(c.form)+'</span>':'');
  const rank='<div class="rank">'+c.ranking.map((r,i)=>'<span class="chip">'+(i+1)+'. <b>'+esc(name(r.reciter))+'</b> '
    +(c.kind=='letter'?(r.median_weakest??'—')+' · '+Math.round(100*r.clean)+'% clean':Math.round(100*r.passed)+'% pass'+(r.median_abs_z!=null?' · |z| '+r.median_abs_z:''))+'</span>').join('')+'</div>';
  const order=c.ranking.map(r=>r.reciter);
  const rows=order.map(rec=>'<div class="row"><div class="who">'+esc(name(rec))+(C.fast.includes(rec)?'<span class="tag">fast (ḥadr)</span>':'')+'</div><div><div class="clips">'
     +c.reciters[rec].map((x,i)=>inst(c,x,rec,i)).join('')+'</div><div class="det" id="d-'+esc(c.cell+rec)+'"></div></div></div>').join('');
  return '<div class="card"><h2>'+title+'</h2>'+rank+rows+'</div>';
}
function inst(c,x,rec,i){
  const v=c.kind=='letter'?(x.weakest==null?'<span class="mut">—</span>':'<span class="'+(x.clean?'ok':'bad')+'">'+x.weakest+'</span>')
     :'<span class="'+(x.status=='pass'?'ok':'bad')+'">'+esc(x.status)+(x.counts!=null?' '+x.counts:'')+'</span>';
  return '<div class="clip" data-c="'+esc(c.cell)+'" data-r="'+esc(rec)+'" data-i="'+i+'" onclick="play(this)"><span class="w">'+esc(x.word)+'</span>▶ '+v+'</div>';
}
function detail(c,x){
  if(c.kind=='rule')return esc(x.ref)+' · '+esc(x.status)+' · counts '+(x.counts??'—')+(x.expected?' (band '+x.expected.join('–')+')':'')+' · z vs masters '+(x.z_masters??'—')+' · '+x.seconds+' s';
  const ch=Object.entries(x.checks).map(([k,v])=>k+' '+v).join(', ');
  const mk=Object.entries(x.makhraj||{}).map(([q,v])=>q+' '+v).join(', ');
  return esc(x.ref)+' · '+(x.clean?'every check held':'NOT held: '+esc(Object.entries(x.heard).map(([h,o])=>h+' heard '+o).join(', ')+(x.identity_heard?' identity heard '+x.identity_heard:'')))
    +'\\nweakest: '+esc(x.weakest_check)+' '+x.weakest+'\\nchecks (margin, nats): '+esc(ch)+(mk?'\\nmakhraj point '+x.makhraj_point+' against its neighbours: '+esc(mk):'')+'\\n'+x.duration_s+' s';
}
function play(el){
  const c=C.cells.find(k=>k.cell==el.dataset.c),x=c.reciters[el.dataset.r][+el.dataset.i];
  document.getElementById('d-'+el.dataset.c+el.dataset.r).textContent=detail(c,x);
  document.querySelectorAll('.clip.playing').forEach(e=>e.classList.remove('playing'));el.classList.add('playing');
  if(cur)cur.pause();cur=new Audio('/letters/audio/'+encodeURI(RAW?x.clip.replace(/\\.wav$/,'.raw.wav'):x.clip));
  cur.onended=()=>el.classList.remove('playing');cur.play();
}
function best(){
  const recs=C.reciters, win={}, n={};
  recs.forEach(r=>{win[r]=0;n[r]=0;});
  C.cells.forEach(c=>{c.ranking.forEach((r,i)=>{n[r.reciter]++;if(i==0)win[r.reciter]++;});});
  const byLetter=C.cells.filter(c=>c.kind=='letter');
  let h='<div class="card"><h2>Cells where each reciter reads best</h2><p class="mut">Letters: highest median weakest margin (the check each instance came closest to failing). Rules: most passes, then nearest the masters.</p><table><tr><th>reciter</th><th>best in</th><th>of cells read</th></tr>'
    +recs.map(r=>'<tr><td>'+esc(name(r))+(C.fast.includes(r)?' <span class="warn">fast</span>':'')+'</td><td>'+win[r]+'</td><td>'+n[r]+'</td></tr>').join('')+'</table></div>';
  const letters=[...new Set(byLetter.map(c=>c.what))];
  h+='<div class="card"><h2>Per letter: the clearest reader</h2><table><tr><th>letter</th>'+C.forms.map(f=>'<th>'+esc(FORM[f])+'</th>').join('')+'</tr>'
    +letters.map(L=>'<tr><td style="font-size:1.2rem">'+esc(L)+'</td>'+C.forms.map(f=>{const c=byLetter.find(x=>x.what==L&&x.form==f);return '<td>'+(c&&c.ranking[0]?esc(name(c.ranking[0].reciter))+' <span class="mut">'+(c.ranking[0].median_weakest??'')+'</span>':'')+'</td>';}).join('')+'</tr>').join('')+'</table></div>';
  return h;
}
function perturb(){
  if(!P)return '<p class="mut">The perturbation lab has not been run yet.</p>';
  return '<div class="card"><h2>Perturbation lab</h2><p class="mut">'+esc(P.about)+'</p></div>'+P.cases.map(k=>'<div class="card"><h2>'+esc(k.title)+'</h2><p class="mut">'+esc(k.how)+'</p><table><tr><th>version</th><th>play</th><th>engine</th></tr>'
    +k.versions.map(v=>'<tr><td>'+esc(v.label)+'</td><td><button onclick="pplay(\\''+esc(v.clip)+'\\')">▶</button></td><td>'+esc(v.summary)+'</td></tr>').join('')+'</table></div>').join('');
}
function pplay(p){if(cur)cur.pause();cur=new Audio('/letters/audio/'+encodeURI(p));cur.play();}
init();
</script></body></html>
"""
