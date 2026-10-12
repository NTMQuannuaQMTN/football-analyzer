"""Create a private file-based point review; exported labels are evaluation-only."""
import json
from pathlib import Path

import cv2
import numpy as np

from cv.refine_calibration import pitch_markings
from cv.soccernet import Calibration, DEFAULT_GAME, game_dir, on_main_camera


def main():
    root=Path('cv/out/audit8_20261011/manual_review');root.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(game_dir(DEFAULT_GAME)/'1_720p.mkv'))
    if not cap.isOpened():raise FileNotFoundError('Local Arsenal half-1 video missing')
    calibration=Calibration(DEFAULT_GAME,1);frames=[]
    for t in [1444.2,1444.4,1450,1510,1600,1603.2,1603.4,1680,1740,1785,1800,1810,1815,1838,1850,1900,1961.2,1961.4]:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,image=cap.read()
        if not ok:raise RuntimeError(f'Unreadable frame {t}')
        image=cv2.resize(image,(1280,720));name=f'arsenal_{t}.jpg';cv2.imwrite(str(root/name),image)
        h=calibration.at(t);marks=[]
        if h is not None:
            world=pitch_markings(step=2);hp=np.column_stack([world,np.ones(len(world))])@np.linalg.inv(h).T
            points=hp[:,:2]/hp[:,2:]
            marks=[p.tolist() for p in points if np.isfinite(p).all() and 0<=p[0]<1280 and 0<=p[1]<720]
        frames.append(dict(id=f'arsenal_{t}',video_s=t,image=name,main_camera=on_main_camera(DEFAULT_GAME,1,t),projected_marks=marks))
    cap.release()
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Private calibration and identity review</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:16px;background:#f7f8fa;color:#17212b}canvas{width:100%;height:auto;background:#111;cursor:crosshair}button,select,input{font:inherit;padding:7px;margin:5px}table{width:100%;border-collapse:collapse;margin-top:18px}th,td{border:1px solid #ccd3db;padding:8px}small{display:block;margin:8px 0}</style>
<h1>Private camera and identity review</h1><p>Click a clearly visible pitch landmark or a player's ground contact point. These labels are evaluation-only. Keep images and this page local. No server or API is used.</p>
<select id="frame"></select><span id="status"></span><button id="overlay">Show camera paint projection</button>
<canvas id="canvas" width="1280" height="720"></canvas>
<select id="kind"><option value="landmark">Pitch landmark</option><option value="identity">Readable shirt number</option><option value="unclear">Unclear / N/A</option></select>
<select id="landmark"></select><input id="shirt" placeholder="Shirt number, or N/A"><select id="team"><option value="N/A">Team N/A</option><option value="Arsenal">Arsenal</option><option value="Liverpool">Liverpool</option></select><input id="note" placeholder="Optional note">
<small>Left/right refer to the coordinate-system goal ends, not the screen edges. Top means far touchline (y=0); bottom means near touchline (y=68). If orientation, feet or a number is uncertain, use N/A. Elevated goal bars are not pitch landmarks.</small>
<button id="undo">Undo last point</button><button id="export">Export review CSV</button><p id="message"></p>
<table><thead><tr><th>Frame</th><th>Kind</th><th>Label</th><th>Image point</th><th>Pitch point</th><th>Note</th></tr></thead><tbody id="rows"></tbody></table>
<script>
const frames=__FRAMES__, landmarks={"Left far corner":[0,0],"Left near corner":[0,68],"Right far corner":[105,0],"Right near corner":[105,68],"Halfway far intersection":[52.5,0],"Halfway near intersection":[52.5,68],"Left penalty-area far front corner":[16.5,13.84],"Left penalty-area near front corner":[16.5,54.16],"Right penalty-area far front corner":[88.5,13.84],"Right penalty-area near front corner":[88.5,54.16],"Left six-yard far front corner":[5.5,24.84],"Left six-yard near front corner":[5.5,43.16],"Right six-yard far front corner":[99.5,24.84],"Right six-yard near front corner":[99.5,43.16]};
const $=id=>document.getElementById(id), canvas=$('canvas'),ctx=canvas.getContext('2d');let img=new Image(),show=false,labels=[];
try{labels=JSON.parse(localStorage.getItem('football_audit8_manual_labels')||'[]')}catch(e){}
frames.forEach((f,i)=>$('frame').add(new Option(`${f.video_s}s · ${f.main_camera?'main camera':'replay/close-up'}`,i)));Object.keys(landmarks).forEach(n=>$('landmark').add(new Option(n,n)));
const current=()=>frames[Number($('frame').value)];
function draw(){ctx.drawImage(img,0,0,1280,720);if(show){ctx.fillStyle='#ffd65c';for(const p of current().projected_marks){ctx.beginPath();ctx.arc(p[0],p[1],2,0,Math.PI*2);ctx.fill()}}for(const r of labels.filter(x=>x.frame_id===current().id)){ctx.fillStyle='#00d8ff';ctx.beginPath();ctx.arc(r.pixel_x,r.pixel_y,5,0,Math.PI*2);ctx.fill();ctx.fillText(r.label,r.pixel_x+7,r.pixel_y-7)}}
function rows(){const body=$('rows');body.replaceChildren();for(const r of labels){const tr=document.createElement('tr');for(const value of [r.frame_id,r.kind,r.kind==='identity'?`${r.team||'N/A'} ${r.label}`:r.label,`${r.pixel_x}, ${r.pixel_y}`,r.world_x===''?'N/A':`${r.world_x}, ${r.world_y}`,r.notes]){const td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}try{localStorage.setItem('football_audit8_manual_labels',JSON.stringify(labels))}catch(e){$('message').textContent='Local storage unavailable: export before closing.'}}
function load(){const f=current();$('status').textContent=f.main_camera?' Main camera: eligible for geometry review':' Replay/close-up: do not compare with current match positions';img.onload=draw;img.src=f.image}
$('frame').onchange=load;$('overlay').onclick=()=>{show=!show;$('overlay').textContent=show?'Hide camera paint projection':'Show camera paint projection';draw()};
canvas.onclick=e=>{const f=current(),kind=$('kind').value;let label=kind==='landmark'?$('landmark').value:($('shirt').value.trim()||'N/A'),world=kind==='landmark'?landmarks[label]:['',''];const rect=canvas.getBoundingClientRect();labels.push({frame_id:f.id,video_s:f.video_s,kind,label,team:kind==='identity'?$('team').value:'N/A',pixel_x:Math.round((e.clientX-rect.left)*1280/rect.width),pixel_y:Math.round((e.clientY-rect.top)*720/rect.height),world_x:world[0],world_y:world[1],notes:$('note').value.trim()});rows();draw()};
$('undo').onclick=()=>{labels.pop();rows();draw()};$('export').onclick=()=>{const keys=['frame_id','video_s','kind','label','team','pixel_x','pixel_y','world_x','world_y','notes'];const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const csv=[keys.join(','),...labels.map(r=>keys.map(k=>quote(r[k])).join(','))].join(String.fromCharCode(10));const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='arsenal_manual_review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('message').textContent='Copy the exported CSV into cv/out/audit8_20261011/manual_review/ for evaluation. Images stay private.'};rows();load();
</script></html>'''
    (root/'review.html').write_text(template.replace('__FRAMES__',json.dumps(frames)))
    (root/'frames.json').write_text(json.dumps(frames,indent=2));print(root/'review.html')


if __name__=='__main__':main()
