"""HTML for the device dashboard, kept separate from the HTTP plumbing."""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Neurix · Durian AI</title>
  <style>
    :root{--bg:#07131d;--panel:#0d2130;--panel-soft:#102838;--line:#274657;--text:#f7fbfd;--muted:#9fb5c2;
      --orange:#f37021;--orange-soft:#ff9b57;--blue:#1686c9;--green:#22a958;--yellow:#ffd166;--red:#ff6b6b}
    *{box-sizing:border-box}body{margin:0;background:
      radial-gradient(circle at 12% -10%,rgba(243,112,33,.22),transparent 34%),
      radial-gradient(circle at 88% 0,rgba(22,134,201,.18),transparent 30%),
      linear-gradient(160deg,#0a1b27 0,var(--bg) 58%,#071822 100%);
      color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh}
    body:before{content:"";display:block;height:5px;background:linear-gradient(90deg,var(--orange) 0 50%,var(--blue) 50% 76%,var(--green) 76%)}
    header{max-width:1280px;margin:auto;padding:22px 18px 10px;display:flex;justify-content:space-between;
      align-items:center;gap:16px;flex-wrap:wrap}h1{font-size:clamp(22px,4vw,34px);margin:0;letter-spacing:-.03em}
    .header-identity{display:flex;align-items:center;gap:20px;min-width:0}.header-logos{display:block;width:clamp(280px,34vw,430px);height:auto;
      max-height:126px;object-fit:contain}.header-copy{min-width:240px}.brand{color:var(--text);font-weight:850}.brand:after{content:"";display:inline-block;width:8px;height:8px;
      margin-left:7px;border-radius:50%;background:var(--orange);box-shadow:12px 0 var(--blue),24px 0 var(--green)}
    .sub{color:var(--muted);margin-top:3px}.pill{padding:8px 13px;border:1px solid rgba(243,112,33,.48);
      border-radius:999px;background:rgba(243,112,33,.1);font-weight:750;box-shadow:inset 0 0 0 1px #0002}
    main{max-width:1280px;margin:auto;padding:10px 18px 32px;display:grid;gap:16px}
    .grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(310px,.75fr);gap:16px}
    .panel{background:linear-gradient(145deg,rgba(16,40,56,.96),rgba(9,27,40,.98));border:1px solid var(--line);
      border-radius:18px;padding:16px;box-shadow:0 18px 50px #0006,0 1px 0 rgba(255,255,255,.035) inset}
    h2{font-size:17px;margin:0 0 12px;display:flex;align-items:center;gap:9px}h2:before{content:"";width:4px;height:18px;
      border-radius:4px;background:var(--orange);box-shadow:0 0 14px rgba(243,112,33,.35)}
    .stage{position:relative;display:grid;place-items:center;background:#03090d;border:1px solid #294452;border-radius:12px;overflow:hidden;min-height:300px}
    #preview{display:block;width:auto;height:auto;max-width:100%;max-height:620px}#roi{position:absolute;left:0;top:0;width:1px;height:1px;
      cursor:crosshair;touch-action:none;background:transparent}.tools{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center}
    button{border:1px solid #365a6c;background:#17384a;color:#fff;border-radius:9px;padding:8px 11px;font-weight:700;cursor:pointer;
      transition:background .18s,border-color .18s,transform .18s}button:hover{background:#21516b;border-color:#4f7b91;transform:translateY(-1px)}
    #reset,.rotate.active{background:var(--orange);border-color:var(--orange-soft);color:#fff}#reset:hover,.rotate.active:hover{background:#d95d15}
    .tool-status{color:var(--muted);margin-left:auto;font-size:13px}
    .result-hero{padding:20px;border-radius:14px;background:linear-gradient(145deg,rgba(243,112,33,.13),rgba(22,134,201,.09));
      border:1px solid rgba(243,112,33,.38);text-align:center}
    .label{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
    .answer{font-size:clamp(30px,6vw,54px);font-weight:900;margin:5px 0;color:var(--orange)}
    .confidence{font-size:18px;font-weight:700}.message{margin-top:13px;padding:11px;border-left:3px solid var(--orange);
      border-radius:10px;background:rgba(243,112,33,.1);color:#fff3e9}
    .meta{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.metric{padding:10px;background:rgba(4,17,25,.55);border:1px solid #1e3c4c;border-radius:10px}
    .metric strong{display:block;margin-top:3px}.bars{display:grid;gap:9px;margin-top:15px}.bar-row{display:grid;grid-template-columns:78px 1fr 48px;gap:8px;align-items:center}
    .track{height:9px;border-radius:9px;background:#203946;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--orange),#ffad4d)}
    .shots{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.shot{background:rgba(5,20,30,.7);border:1px solid var(--line);border-radius:13px;overflow:hidden}
    .shot img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#03090d}.shot-info{padding:10px;font-size:13px;display:flex;align-items:center;justify-content:space-between;gap:10px}
    .shot-title{font-weight:800}.muted{color:var(--muted)}.empty{display:grid;place-items:center;min-height:160px;color:var(--muted)}
    audio{width:100%;margin-top:9px}.audio-file{width:100%}
    .status-processing{color:var(--yellow)}.status-error,.status-no_durian{color:var(--red)}.status-completed{color:#73dc94;border-color:rgba(34,169,88,.55);background:rgba(34,169,88,.1)}
    @media(max-width:850px){.grid{grid-template-columns:1fr}.shots{grid-template-columns:1fr}.stage{min-height:220px}.header-identity{width:100%;flex-direction:column;align-items:flex-start;gap:8px}.header-logos{width:min(100%,390px)}.header-copy{min-width:0}}
  </style>
</head>
<body>
  <header><div class="header-identity"><img class="header-logos" src="/assets/header-logos.png" alt="Logo Đại học FPT và Team Neurix"><div class="header-copy"><h1>Hệ thống đánh giá độ chín sầu riêng</h1><div class="sub">Đại học FPT Cần Thơ · <span class="brand">Team Neurix</span></div></div></div>
    <div id="connection" class="pill">Đang kết nối…</div></header>
  <main>
    <section class="grid">
      <div class="panel"><h2>Camera trực tiếp & vùng nhận diện</h2><div class="stage"><img id="preview" alt="Camera"><canvas id="roi"></canvas></div>
        <div class="tools"><button id="reset" type="button">Toàn khung</button><button class="rotate" data-angle="0">0°</button><button class="rotate active" data-angle="90">90°</button><button class="rotate" data-angle="180">180°</button><button class="rotate" data-angle="270">270°</button><span id="roi-status" class="tool-status">Đang tải ROI…</span></div></div>
      <aside class="panel"><h2>Kết quả gần nhất</h2><div class="result-hero"><div class="label">Kết luận</div><div id="answer" class="answer">CHỜ</div><div id="confidence" class="confidence">—</div></div>
        <div id="message" class="message">Sẵn sàng kiểm tra</div><div class="meta"><div class="metric"><span class="muted">Mã lượt</span><strong id="job">—</strong></div><div class="metric"><span class="muted">Tổng thời gian</span><strong id="duration">—</strong></div></div><div id="final-bars" class="bars"></div></aside>
    </section>
    <section class="panel"><h2>Ảnh vừa chụp</h2><div id="shots" class="shots"><div class="empty">Chưa có ảnh</div></div></section>
    <section class="panel"><h2>Âm thanh vừa thu</h2><div id="audio-area" class="empty">Chưa có file âm thanh</div></section>
  </main>
  <script>
    const labels={unripe:'CHƯA CHÍN',ripe:'CHÍN',Overripe:'QUÁ CHÍN',Unknown:'KHÔNG RÕ'};
    const statusLabels={idle:'SẴN SÀNG',processing:'ĐANG XỬ LÝ',capturing:'ĐANG CHỤP',recording:'ĐANG THU ÂM',analyzing:'ĐANG PHÂN TÍCH',completed:'HOÀN TẤT',no_durian:'KHÔNG CÓ SẦU RIÊNG',error:'LỖI'};
    let lastJob=null,lastAudioUrl='';
    const pct=value=>`${(Number(value||0)*100).toFixed(1)}%`;
    function captureTime(value){const date=new Date(Number(value)*1000);return Number.isNaN(date.getTime())?'—':date.toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}
    function bars(result){if(!result)return '';return Object.entries(result.probabilities||{}).map(([name,value])=>`<div class="bar-row"><span>${labels[name]||name}</span><div class="track"><div class="fill" style="width:${pct(value)}"></div></div><strong>${pct(value)}</strong></div>`).join('')}
    function resultText(result){return result?`${labels[result.class_name]||result.class_name} · ${pct(result.confidence)}`:'Chưa có kết quả'}
    function render(data){document.getElementById('connection').textContent=statusLabels[data.status]||data.status;
      document.getElementById('connection').className='pill status-'+data.status;document.getElementById('message').textContent=data.message;
      document.getElementById('job').textContent=data.job_id?`#${data.job_id}`:'—';document.getElementById('duration').textContent=data.duration_ms!=null?`${(data.duration_ms/1000).toFixed(2)} giây`:'—';
      const final=data.final_result,answer=document.getElementById('answer');answer.textContent=final?(labels[final.class_name]||final.class_name):(data.status==='no_durian'?'KHÔNG CÓ QUẢ':(statusLabels[data.status]||'CHỜ'));
      answer.style.color=data.status==='error'||data.status==='no_durian'?'var(--red)':(final?'var(--orange)':'var(--yellow)');document.getElementById('confidence').textContent=final?`Độ tin cậy ${pct(final.confidence)}`:'—';document.getElementById('final-bars').innerHTML=bars(final);
      const shots=document.getElementById('shots');shots.innerHTML=data.images.length?data.images.map((item,index)=>`<article class="shot"><img src="${item.url}?job=${data.job_id}&v=${data.updated_at}" alt="Ảnh ${index+1}"><div class="shot-info"><div class="shot-title">Ảnh ${index+1}</div><div class="muted">Chụp lúc ${captureTime(item.captured_at)}</div></div></article>`).join(''):'<div class="empty">Chưa có ảnh</div>';
      const area=document.getElementById('audio-area');if(data.audio){const url=`${data.audio.url}?job=${data.job_id}`;if(lastAudioUrl!==url){area.className='audio-file';area.innerHTML=`<div class="muted">${data.audio.name}</div><audio controls preload="metadata" src="${url}"></audio>`;lastAudioUrl=url}}else if(lastAudioUrl){area.className='empty';area.textContent='Chưa có file âm thanh';lastAudioUrl=''}lastJob=data.job_id}
    async function refresh(){try{const response=await fetch('/api/result',{cache:'no-store'});if(!response.ok)throw Error();render(await response.json())}catch{const el=document.getElementById('connection');el.textContent='MẤT KẾT NỐI';el.className='pill status-error'}}setInterval(refresh,1000);refresh();

    const image=document.getElementById('preview'),canvas=document.getElementById('roi'),stage=canvas.parentElement,ctx=canvas.getContext('2d'),roiStatus=document.getElementById('roi-status');let start=null,draft=null,resizeObserver=null,rotation=localStorage.getItem('previewRotation')||'90';
    if(!['0','90','180','270'].includes(rotation))rotation='90';
    function displayedToRaw(v){if(rotation==='90')return{x:v.y,y:1-v.x-v.width,width:v.height,height:v.width};if(rotation==='180')return{x:1-v.x-v.width,y:1-v.y-v.height,width:v.width,height:v.height};if(rotation==='270')return{x:1-v.y-v.height,y:v.x,width:v.height,height:v.width};return v}
    function resizeCanvas(){const w=Math.max(1,image.offsetWidth),h=Math.max(1,image.offsetHeight);canvas.style.left=image.offsetLeft+'px';canvas.style.top=image.offsetTop+'px';canvas.style.width=w+'px';canvas.style.height=h+'px';if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}draw(draft)}
    function draw(v){ctx.clearRect(0,0,canvas.width,canvas.height);if(!v)return;ctx.strokeStyle='#ff8a1f';ctx.lineWidth=3;ctx.setLineDash([10,6]);ctx.strokeRect(v.x*canvas.width,v.y*canvas.height,v.width*canvas.width,v.height*canvas.height);ctx.setLineDash([])}
    function point(e){const b=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)),y:Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))}}
    async function save(v){roiStatus.textContent='Đang lưu…';const r=await fetch('/api/roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(v)});const p=await r.json();if(!r.ok)throw Error(p.error||'Không lưu được ROI');draft=null;draw(null);roiStatus.textContent='Đã lưu vùng nhận diện'}
    canvas.addEventListener('pointerdown',e=>{start=point(e);draft={x:start.x,y:start.y,width:0,height:0};canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!start)return;const c=point(e),l=Math.min(start.x,c.x),t=Math.min(start.y,c.y);draft={x:l,y:t,width:Math.abs(c.x-start.x),height:Math.abs(c.y-start.y)};draw(draft)});canvas.addEventListener('pointerup',async e=>{if(!start||!draft)return;start=null;canvas.releasePointerCapture(e.pointerId);if(draft.width<.02||draft.height<.02){draft=null;draw(null);return}try{await save(displayedToRaw(draft))}catch(err){roiStatus.textContent=err.message}});canvas.addEventListener('pointercancel',()=>{start=null;draft=null;draw(null)});
    function setRotation(angle){rotation=angle;localStorage.setItem('previewRotation',angle);document.querySelectorAll('.rotate').forEach(b=>b.classList.toggle('active',b.dataset.angle===angle));roiStatus.textContent='Đang xoay camera…';image.src='/stream.mjpg?rotate='+angle+'&t='+Date.now()}
    document.getElementById('reset').onclick=()=>save({x:0,y:0,width:1,height:1});document.querySelectorAll('.rotate').forEach(b=>b.onclick=()=>setRotation(b.dataset.angle));image.addEventListener('load',resizeCanvas);window.addEventListener('resize',resizeCanvas);if(window.ResizeObserver){resizeObserver=new ResizeObserver(()=>requestAnimationFrame(resizeCanvas));resizeObserver.observe(stage);resizeObserver.observe(image)}fetch('/api/roi').then(r=>r.json()).then(()=>{roiStatus.textContent='Kéo trên ảnh để đổi ROI';resizeCanvas()});setRotation(rotation);
  </script>
</body>
</html>"""
