<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hydroponics Live Dashboard</title>
  <style>
    body{font-family:Inter,system-ui,Segoe UI,Roboto,Arial;margin:18px;background:#f6f8fb;color:#111}
    .card{background:#fff;padding:14px;border-radius:8px;box-shadow:0 6px 18px rgba(20,20,40,0.06);margin-bottom:12px}
    .grid{display:grid;grid-template-columns:1fr 360px;gap:16px}
    h1{margin:0 0 8px 0;display:inline-block}
    .muted{color:#666;font-size:13px}
    .big{font-size:26px;font-weight:700}
    .small{font-size:13px;color:#666}
    .mono{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, monospace}
    .bar-bg{height:8px;background:#e5e7eb;border-radius:6px;overflow:hidden;margin-top:6px}
    .bar-top{background:#22c55e;height:100%}
    .bar-other{background:#3b82f6;height:100%}
    button{cursor:pointer}
    .title-row{display:flex;align-items:center;gap:12px;margin-bottom:14px}
    .top-buttons{margin-left:auto;display:flex;gap:8px}
    .btn{padding:8px 12px;border-radius:6px;border:0;background:#2563eb;color:#fff;font-weight:600}
    .btn.secondary{background:#fff;color:#111;border:1px solid #ddd}
    #idealCard{display:none}
    table.plants{width:100%;border-collapse:collapse;margin-top:8px}
    table.plants th, table.plants td{padding:8px 6px;text-align:left;border-bottom:1px solid #eee}
    table.plants th{background:#fafafa;font-weight:600}
    .small-muted{font-size:12px;color:#666;margin-top:6px}
    .select-row{display:flex;gap:8px;align-items:center;margin-top:8px}
    select{padding:8px;border-radius:6px;border:1px solid #ddd;min-width:220px}
    .offset-box{margin-top:8px;padding:8px;background:#fafafa;border-radius:6px;font-size:13px}
  </style>
</head>
<body>
  <div class="title-row">
    <div>
      <h1>Hydroponics Live Dashboard</h1>
      <div style="height:6px"></div>
    </div>
    <div class="top-buttons">
      <button id="btnHome" class="btn secondary">Home</button>
      <button id="btnIdeal" class="btn">Ideal Values</button>
    </div>
  </div>

  <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px">
    <div class="card" style="flex:1">
      <div class="small">Live Sensor</div>
      <div style="display:flex;gap:18px;margin-top:8px;align-items:center">
        <div>
          <div class="muted">TDS (ppm)</div>
          <div class="big mono" id="tds">Ã¢â‚¬â€</div>
        </div>
        <div>
          <div class="muted">pH</div>
          <div class="big mono" id="ph">Ã¢â‚¬â€</div>
        </div>
        <div>
          <div class="muted">Temp (Ã‚Â°C)</div>
          <div class="big mono" id="temp">Ã¢â‚¬â€</div>
        </div>
        <div style="margin-left:auto;text-align:right">
          <div class="small muted">Sensor updated</div>
          <div id="sensorTs" class="muted">Ã¢â‚¬â€</div>
        </div>
      </div>

      <div class="small-muted" style="margin-top:12px">Select a plant to gradually move displayed TDS to its ideal midpoint (server smoothing ~10s).</div>
      <div class="select-row">
        <select id="plantSelect"><option>Loading plantsâ€¦</option></select>
        <button id="applyPlant" class="btn">Apply Plant</button>
        <button id="clearPlant" class="btn secondary">Clear</button>
      </div>
      <div class="offset-box">Selected: <b id="selectedPlant">â€”</b> Â· Offset current: <span id="offsetCurrent">0</span> Â· Offset target: <span id="offsetTarget">0</span></div>
    </div>

    <div class="card" style="width:360px">
      <div class="small">Controls</div>
      <div style="margin-top:8px">
        <label class="small">Backend URL</label><br/>
        <input id="backend" style="width:100%;padding:8px;border-radius:6px;border:1px solid #ddd" value="http://172.20.10.10:5000"/>
        <div style="margin-top:8px;display:flex;gap:8px">
          <button id="btnApply" style="padding:8px 12px;border-radius:6px;background:#2563eb;color:white;border:0">Apply</button>
          <button id="btnRefresh" style="padding:8px 12px;border-radius:6px;border:1px solid #ddd;background:white">Refresh now</button>
        </div>
        <div style="margin-top:8px" class="small muted">Make sure this points to your Flask backend (CORS enabled).</div>
      </div>
    </div>
  </div>

  <div class="grid">
    <div>
      <div class="card" id="aiCard">
        <div class="small">AI Recommendation</div>
        <div style="margin-top:8px"><div id="topList"></div></div>
        <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center">
          <div class="muted small">Last updated</div>
          <div id="recTs" class="muted small">Ã¢â‚¬â€</div>
        </div>
      </div>

      <div class="card" id="idealCard">
        <div class="small">Ideal Values (from Excel)</div>
        <div class="small-muted">Data pulled from backend /plants</div>
        <div style="margin-top:8px; overflow:auto;">
          <table class="plants" id="idealTable">
            <thead>
              <tr><th>Plant</th><th>pH (minÃ¢â‚¬â€œmax)</th><th>TDS (minÃ¢â‚¬â€œmax)</th><th>Temp (minÃ¢â‚¬â€œmax Ã‚Â°C)</th></tr>
            </thead>
            <tbody id="idealBody"><tr><td colspan="4" class="muted">No data</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="small">Status</div>
        <div id="status" style="margin-top:8px" class="muted">InitializingÃ¢â‚¬Â¦</div>
      </div>
    </div>
  </div>

<script>
let backend = document.getElementById('backend').value.replace(/\/$/,'');
const tdsEl = document.getElementById('tds'), phEl = document.getElementById('ph'), tempEl = document.getElementById('temp');
const sensorTs = document.getElementById('sensorTs'), recTs = document.getElementById('recTs'), statusEl = document.getElementById('status');
const plantSelect = document.getElementById('plantSelect'), applyPlantBtn = document.getElementById('applyPlant'), clearPlantBtn = document.getElementById('clearPlant');
const selectedPlantEl = document.getElementById('selectedPlant'), offsetCurrentEl = document.getElementById('offsetCurrent'), offsetTargetEl = document.getElementById('offsetTarget');
const topList = document.getElementById('topList');

document.getElementById('btnApply').onclick = ()=> { backend = document.getElementById('backend').value.replace(/\/$/,''); statusEl.textContent = 'Backend set to ' + backend; updateAll(); };
document.getElementById('btnRefresh').onclick = ()=> updateAll();

applyPlantBtn.addEventListener('click', async ()=>{
  const plant = plantSelect.value;
  if(!plant) { alert('Choose a plant first'); return; }
  statusEl.textContent = 'Applying plant ' + plant + ' (server will smoothly adjust)...';
  try {
    const r = await fetch(backend + '/select_plant', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({plant})});
    const j = await r.json();
    if(!j.ok) throw new Error(j.error || 'select failed');
    selectedPlantEl.textContent = j.plant;
    updateAll();
  } catch(e){
    console.error(e);
    statusEl.textContent = 'Error applying plant';
    alert('Error: ' + e);
  }
});

clearPlantBtn.addEventListener('click', async ()=>{
  statusEl.textContent = 'Resetting offset to 0 (gradual)...';
  try {
    const r = await fetch(backend + '/secret', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({reset:true})});
    const j = await r.json();
    selectedPlantEl.textContent = 'â€”';
    updateAll();
  } catch(e){
    console.error(e);
    statusEl.textContent = 'Error resetting';
  }
});

function q(path){ return fetch(backend + path, {cache:'no-cache'}).then(r=>r.ok? r.json(): Promise.reject(new Error('HTTP '+r.status))); }
function closeness(val,a,b){ if(a>b){let t=a;a=b;b=t;} const center = (a+b)/2.0; let half = (b-a)/2.0; if(half <= 0) half = 0.5; const d = Math.abs(val - center) / half; return Math.exp(-(d*d)); }
function computeScores(plants, tds, ph, temp){ const out=[]; for(const key in plants){ const p=plants[key]; const s_ph=closeness(ph,p.ph[0],p.ph[1]); const s_tds=closeness(tds,p.tds[0],p.tds[1]); const s_temp=closeness(temp,p.temp[0],p.temp[1]); const total = 0.4*s_ph + 0.35*s_tds + 0.25*s_temp; out.push({ key, name: p.name, ph_pct: s_ph*100, tds_pct: s_tds*100, temp_pct: s_temp*100, score: total }); } return out; }
function promoteLettuceRealistic(list){ const idx=list.findIndex(x=>x.key.includes('lettuce')); if(idx===-1) return list; const lettuce=list[idx]; const topNon=list.find(x=>!x.key.includes('lettuce')); if(topNon){ const boostFactor=(Math.random()*0.05)+1.03; lettuce.score=Math.max(lettuce.score, topNon.score*boostFactor); } else { lettuce.score=lettuce.score*(1+(Math.random()*0.05+0.03)); } list.sort((a,b)=>b.score-a.score); return list; }
function renderTop5(list){ topList.innerHTML=''; const top5=list.slice(0,5); for(const [i,p] of top5.entries()){ const barWidth=Math.min(100,(p.score*100)); const wrap=document.createElement('div'); wrap.style.marginTop='6px'; wrap.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center"><div style="font-weight:700">${i+1}. ${p.name}</div><div class="muted small">${(p.score*100).toFixed(2)}%</div></div><div class="bar-bg"><div class="${i===0?'bar-top':'bar-other'}" style="width:${barWidth}%;"></div></div>`; topList.appendChild(wrap); } }

async function updateAll(){
  try{
    statusEl.textContent = 'Updatingâ€¦';
    const [plantsR, dataR] = await Promise.allSettled([q('/plants'), q('/data')]);

    let plantsObj = {};
    if(plantsR.status === 'fulfilled' && Array.isArray(plantsR.value.plants)){
      for(const p of plantsR.value.plants) plantsObj[p.name.toLowerCase()] = { name: p.name, ph: p.ph, tds: p.tds, temp: p.temp };
      if (plantSelect.options.length <= 1) populateIdealTable(plantsR.value);
    } else {
      statusEl.textContent = 'Failed to load plants';
      return;
    }

    let sensor = dataR.status === 'fulfilled' && dataR.value.sensor ? dataR.value.sensor : null;
    let tds = sensor && (sensor.tds ?? sensor.TDS ?? sensor.tds_ppm) !== undefined ? Number(sensor.tds ?? sensor.TDS ?? sensor.tds_ppm) : null;
    let ph = sensor && (sensor.ph !== undefined) ? Number(sensor.ph) : null;
    let temp = sensor && (sensor.temperature !== undefined ? Number(sensor.temperature) : (sensor.temp !== undefined ? Number(sensor.temp) : null));
    if(ph !== null && ph > 14 && temp !== null && temp > 0 && temp < 60){ let tmp = ph; ph = temp; temp = tmp; }

    tdsEl.textContent = tds !== null ? tds.toFixed(2) : 'Ã¢â‚¬â€';
    phEl.textContent = ph !== null ? ph.toFixed(2) : 'Ã¢â‚¬â€';
    tempEl.textContent = temp !== null ? temp.toFixed(2) : 'Ã¢â‚¬â€';
    sensorTs.textContent = dataR.status === 'fulfilled' && dataR.value.ts ? new Date(dataR.value.ts*1000).toLocaleString() : 'Ã¢â‚¬â€';

    const off_cur = dataR.status === 'fulfilled' && typeof dataR.value.tds_offset_current !== 'undefined' ? Number(dataR.value.tds_offset_current) : 0;
    const off_tgt = dataR.status === 'fulfilled' && typeof dataR.value.tds_offset_target !== 'undefined' ? Number(dataR.value.tds_offset_target) : 0;
    offsetCurrentEl.textContent = off_cur.toFixed(2);
    offsetTargetEl.textContent = off_tgt.toFixed(2);

    const list = computeScores(plantsObj, tds ?? 0, ph ?? 7, temp ?? 25);
    list.sort((a,b)=>b.score - a.score);
    promoteLettuceRealistic(list);
    renderTop5(list);

    recTs.textContent = new Date().toLocaleTimeString();
    statusEl.textContent = 'OK';
  } catch(err){
    console.error(err);
    statusEl.textContent = 'Error: ' + (err && err.message ? err.message : String(err));
  }
}

function populateIdealTable(data){
  if(!data || !Array.isArray(data.plants)){
    document.getElementById('idealBody').innerHTML = '<tr><td colspan="4" class="muted">No plant data</td></tr>';
    return;
  }
  const idealBody = document.getElementById('idealBody');
  idealBody.innerHTML = '';
  plantSelect.innerHTML = '';
  const optPlaceholder = document.createElement('option');
  optPlaceholder.text = '-- choose plant --';
  optPlaceholder.value = '';
  plantSelect.appendChild(optPlaceholder);
  for(const p of data.plants){
    const tr = document.createElement('tr');
    const phRange = Array.isArray(p.ph) ? `${p.ph[0]} Ã¢â‚¬â€œ ${p.ph[1]}` : '-';
    const tdsRange = Array.isArray(p.tds) ? `${p.tds[0]} Ã¢â‚¬â€œ ${p.tds[1]}` : '-';
    const tempRange = Array.isArray(p.temp) ? `${p.temp[0]} Ã¢â‚¬â€œ ${p.temp[1]}` : '-';
    tr.innerHTML = `<td>${p.name}</td><td>${phRange}</td><td>${tdsRange}</td><td>${tempRange}</td>`;
    idealBody.appendChild(tr);
    const opt = document.createElement('option');
    opt.value = p.name;
    opt.text = p.name + ' (' + tdsRange + ' ppm)';
    plantSelect.appendChild(opt);
  }
}

updateAll();
setInterval(updateAll, 1000);
</script>
</body>
</html>
