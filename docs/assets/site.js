(() => {
'use strict';
const data = JSON.parse(document.getElementById('research-data').textContent);
const $ = id => document.getElementById(id);
const fmt = x => Number(x).toFixed(2) + '%';
let protocol = 'record_models', metric = 'accuracy', selected = null, neuralMetric = 'accuracy';
const friendly = {record_models:'Record-level',window_models:'Within-record windows',unseen_window_models:'Unseen recordings'};
function modelRows(){return [...data[protocol]].sort((a,b)=>b[metric]-a[metric]);}
function chooseModel(name){selected=name;renderModels();}
function renderModels(){
 const rows=modelRows();
 if(!rows.length){$('model-chart').textContent='No model data';return;}
 if(!rows.some(r=>r.model===selected)) selected=rows.find(r=>r.model.includes('Hard Voting'))?.model||rows[0].model;
 $('plot-title').textContent=friendly[protocol]+' · '+({accuracy:'Accuracy',sensitivity:'Preterm sensitivity',specificity:'Term specificity'}[metric]);
 $('plot-context').textContent=(protocol==='record_models'?'300 recordings':'7,200 windows')+' · 10 folds';
 const chart=$('model-chart');chart.replaceChildren();
 rows.forEach(r=>{
  const btn=document.createElement('button');btn.type='button';btn.className='bar-row'+(r.model===selected?' active':'');
  btn.title=r.model+' · '+metric+' '+fmt(r[metric]);btn.setAttribute('aria-pressed',r.model===selected?'true':'false');
  const name=document.createElement('span');name.className='bar-name';name.textContent=r.model;
  const track=document.createElement('span');track.className='bar-track';
  const fill=document.createElement('span');fill.className='bar-fill';fill.style.width=Math.max(0,Math.min(100,r[metric]))+'%';track.appendChild(fill);
  const value=document.createElement('span');value.className='bar-value';value.textContent=fmt(r[metric]);
  btn.append(name,track,value);btn.addEventListener('click',()=>chooseModel(r.model));chart.appendChild(btn);
 });
 const r=rows.find(r=>r.model===selected);$('selected-name').textContent=r.model;
 $('selected-accuracy').textContent=fmt(r.accuracy);$('selected-sensitivity').textContent=fmt(r.sensitivity);$('selected-specificity').textContent=fmt(r.specificity);
 const m=$('matrix');m.replaceChildren();
 [['TN',r.tn,false],['FP',r.fp,true],['FN',r.fn,true],['TP',r.tp,false]].forEach(([name,val,bad])=>{let e=document.createElement('div');e.className=bad?'bad':'';e.title=name;e.textContent=name+' '+Number(val).toLocaleString();m.appendChild(e);});
 $('chart-caption').textContent=protocol==='record_models'?'Each recording appears in a held-out test fold once.':'Scores are pooled over windows; the protocol determines whether windows from a recording can occur on both sides of a fold.';
}
const labels={accuracy:'Accuracy',sensitivity:'Preterm sensitivity',specificity:'Term specificity'};
document.querySelectorAll('[data-protocol]').forEach(btn=>btn.addEventListener('click',()=>{
 protocol=btn.dataset.protocol;selected=null;
 document.querySelectorAll('[data-protocol]').forEach(b=>{b.classList.toggle('active',b===btn);b.setAttribute('aria-pressed',String(b===btn));});renderModels();}));
$('metric').addEventListener('change',e=>{metric=e.target.value;renderModels()});
function exportCsv(){const rows=data[protocol];const keys=['model','accuracy','sensitivity','specificity','tp','fp','tn','fn'];const csv=[keys.join(','),...rows.map(r=>keys.map(k=>JSON.stringify(r[k]??'')).join(','))].join('\n');const blob=new Blob([csv],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ehg_'+protocol+'.csv';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
$('export-data').addEventListener('click',exportCsv);
function renderNeural(){const rows=data.neural.filter(r=>/^LSTM/.test(r.model));const c=$('neural-chart');c.replaceChildren();const max=neuralMetric==='accuracy'?100:0.55;rows.forEach(r=>{const v=r[neuralMetric];const col=document.createElement('div');col.className='neural-col';const val=document.createElement('div');val.className='nval';val.textContent=neuralMetric==='accuracy'?fmt(v):v.toFixed(4);const bar=document.createElement('div');bar.className='nbar';bar.style.height=Math.round(v/max*170)+'px';bar.title=r.model+': '+val.textContent;const name=document.createElement('div');name.className='nname';name.textContent=r.model.replace('LSTM','')+' units';col.append(val,bar,name);c.appendChild(col)})}
$('neural-metric').addEventListener('change',e=>{neuralMetric=e.target.value;renderNeural()});
// Cohort circle drawn directly from the dataset distribution.
function drawRing(){const s=$('cohort-ring');s.innerHTML='';const ns='http://www.w3.org/2000/svg';const cir=(r,stroke,offset,opacity)=>{let e=document.createElementNS(ns,'circle');for(const [k,v] of Object.entries({cx:125,cy:125,r,fill:'none',stroke,'stroke-width':24,'stroke-dasharray':opacity||'none','stroke-dashoffset':offset||0,transform:'rotate(-90 125 125)'})){e.setAttribute(k,v)}return e};const full=2*Math.PI*84,part=full*38/300;s.append(cir(84,'#1c6e70'));s.append(cir(84,'#acdabb',0,part+' '+(full-part)));const t=document.createElementNS(ns,'text');t.setAttribute('x',125);t.setAttribute('y',119);t.setAttribute('text-anchor','middle');t.setAttribute('font-size',38);t.setAttribute('font-weight',700);t.setAttribute('fill','#203f47');t.textContent='300';s.append(t);let label=document.createElementNS(ns,'text');label.setAttribute('x',125);label.setAttribute('y',144);label.setAttribute('text-anchor','middle');label.setAttribute('font-size',12);label.setAttribute('fill','#607678');label.textContent='RECORDINGS';s.append(label)}
// Animation is decorative. Data samples are from the included processed waveform array.
function drawHero(){const c=$('hero-canvas');if(!c)return;const ctx=c.getContext('2d');const dpr=window.devicePixelRatio||1;const rect=c.getBoundingClientRect();c.width=rect.width*dpr;c.height=rect.height*dpr;ctx.scale(dpr,dpr);const W=rect.width,H=rect.height;let t=0;const wave=[data.trace_a,data.trace_b,data.trace_a.map((x,i)=>.45*x+.25*data.trace_b[(i+38)%data.trace_b.length])];const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
 function frame(){ctx.clearRect(0,0,W,H);const padx=28;const left=padx,right=W-24;for(let i=0;i<3;i++){let center=H*(.21+i*.29);ctx.beginPath();ctx.moveTo(left,center);ctx.lineTo(right,center);ctx.strokeStyle='rgba(211,247,230,.13)';ctx.stroke();ctx.font='10px system-ui';ctx.fillStyle='#8caeae';ctx.fillText('S'+(i+1),left,center-49);ctx.beginPath();wave[i].forEach((y,j)=>{const x=left+j/(wave[i].length-1)*(right-left),v=wave[i][(j+(reduce?0:Math.round(t*.18)))%wave[i].length];const z=center-v*37*(i===0?1:.72);if(j===0)ctx.moveTo(x,z);else ctx.lineTo(x,z)});ctx.strokeStyle=['#dceda9','#8adbcf','#b6d0e6'][i];ctx.lineWidth=1.9;ctx.stroke()}ctx.fillStyle='#8caaa9';ctx.font='10px system-ui';ctx.fillText('750 SAMPLES / WINDOW   •   125-SAMPLE SHIFT',left,H-12);if(!reduce){t+=1;requestAnimationFrame(frame)}}frame()}
let resizeToken;window.addEventListener('resize',()=>{clearTimeout(resizeToken);resizeToken=setTimeout(drawHero,250)});
$('copy-commands').addEventListener('click',async()=>{const text=$('setup-commands').textContent;try{await navigator.clipboard.writeText(text);$('copy-commands').textContent='Copied';setTimeout(()=>$('copy-commands').textContent='Copy',1200)}catch{const range=document.createRange();range.selectNodeContents($('setup-commands'));window.getSelection().removeAllRanges();window.getSelection().addRange(range)}});
window.addEventListener('scroll',()=>{$('scroll-progress').style.width=(100*window.scrollY/Math.max(1,document.documentElement.scrollHeight-window.innerHeight))+'%'},{passive:true});
renderModels();renderNeural();drawRing();drawHero();
})();
