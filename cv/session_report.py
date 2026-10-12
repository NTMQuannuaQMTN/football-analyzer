"""Readable HTML tables for the isolated session's actual evaluation outputs."""
import html
import json
from pathlib import Path


def table(headers,rows):
    cell=lambda x:html.escape(str(x if x is not None else 'N/A'))
    return '<table><thead><tr>'+''.join(f'<th>{cell(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell(v)}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def display_rows(rows):
    """Readable display labels; machine artifact keys and metrics stay unchanged."""
    labels={'exact_context':'Exact context','ball_observed_every_30s':'Ball held up to 30 s',
        'possession_unknown':'Possession N/A',
        'midfield_ball_and_possession_unknown':'Midfield ball + possession N/A',
        'clean_control':'Clean control','robust':'Context-trained',
        'width_m':'Team width','depth_m':'Team depth','defensive_line_m':'Defensive line',
        'up_to_3s':'Up to 3 s','3_to_10s':'Over 3 to 10 s','10_to_30s':'Over 10 to 30 s',
        'over_30s':'Over 30 s','never_seen':'Not yet observed'}
    return [[labels.get(value,value) if isinstance(value,str) else value for value in row] for row in rows]


def generate(root):
    parts=['<h1>Local football training review</h1><p>Held-out tracking reconstruction and visible-image diagnostics. These are separate experiments, not proof of EPL broadcast accuracy. Future sightings enter replay only. Active exports are preserved.</p>']
    read=lambda p:json.loads((root/p).read_text()) if (root/p).exists() else None
    current=read('reconstruction/results.json');old=read('reconstruction/baseline_f0_test.json')
    if current and old:
        rows=[]
        for mode,r in current['selected_test'].items():
            for mask,metrics in r['results'].items():
                a=old[mode]['results'][mask]['model'];b=metrics['model']
                rows.append([mode,mask,b['n'],a['mae_m'],b['mae_m'],round(b['mae_m']-a['mae_m'],2),round(b['median_m'],2),round(b['p90_m'],2),f"{b['radius_coverage_80']:.1%}"])
        parts+=['<h2>Hidden-player errors</h2><p>One held-out DFL match, exact visible IDs/positions, 1 Hz, five training matches and one validation match. Old f0 vs validation-selected v4. Sample counts are correlated player-timestamps, not independent matches. Raw model corrections; speed-limited results are separate.</p>',table(['Mode','Visibility mask','N','Old mean (m)','New mean (m)','Change (m)','New median (m)','New p90 (m)','80% radius coverage'],rows)]
        for mode,r in current['selected_test'].items():
            metrics=r['results']['sc_camera']
            rows=[[name,v['n'],v['mae_m'],round(v['median_m'],2),round(v['p90_m'],2)] for name,v in metrics['by_age'].items() if v]
            parts += [f'<h3>{html.escape(mode)}: time since last sighting</h3>',table(['Age','N','Mean (m)','Median (m)','p90 (m)'],rows)]
            rows=[[name,v['n'],v['mae_m'],round(v['p90_m'],2)] for name,v in metrics['tactical_geometry_absolute_error_m'].items() if v]
            parts += [f'<h3>{html.escape(mode)}: shape accuracy after speed limiting</h3>',table(['Feature','Team-seconds','Mean absolute error (m)','p90 absolute error (m)'],rows)]
            groups=''
            for key,label in [('by_role','Pre-match role'),('by_phase','Ball-in-play status'),('by_confidence','Tracking-domain P(error &lt; 5 m)')]:
                rows=[[name,v['n'],v['mae_m'],round(v['median_m'],2),round(v['p90_m'],2)] for name,v in metrics[key].items() if v]
                groups += f'<h4>{label}</h4>'+table(['Category','N','Mean (m)','Median (m)','p90 (m)'],rows)
            parts += [f'<details><summary>Explore {html.escape(mode)} role, game-state and confidence groups</summary><p>Pre-match roles are coarse and may change during play. Ball-in-play status is not a detailed tactical phase. Confidence categories apply to this tracking benchmark only.</p>{groups}</details>']
    limited=read('reconstruction/postprocess_comparison.json')
    if limited:
        rows=[]
        for mode in ('live','replay'):
            a=limited['baseline'][mode]['results']['sc_camera']['model_speed_limited']
            b=limited['candidate'][mode]['results']['sc_camera']['model_speed_limited']
            rows.append([mode,b['n'],a['mae_m'],b['mae_m'],round(b['p90_m'],2),b['hidden_steps_over_10m_s'],f"{b['radius_coverage_80']:.1%}"])
        parts += ['<h2>Same speed limiter, old versus new model</h2><p>Both models use the existing limiter. Measured sightings are preserved. Goalkeeper-zone and event-cue priors are excluded. Zero hidden-to-hidden steps above 10 m/s does not eliminate identity jumps when a player reappears.</p>',table(['Mode','N','Old mean (m)','New mean (m)','New p90 (m)','Hidden steps over 10 m/s','80% radius coverage'],rows)]
    rows=[]
    for file,label in [('detector_baseline_yolo11s.json','Original generic 11s'),('detector_baseline_yolo11m.json','Archived generic 11m'),('detector_candidate_people.json','Initial two-epoch candidate'),('detector_continuation_people.json','Continued candidate'),('detector_final_people.json','Low-rate final candidate')]:
        r=read(file)
        if r: rows.append([label,r['frames'],r['gt_people'],r['matched_iou_50'],f"{r['precision']:.1%}",f"{r['recall']:.1%}",r['oracle_calibrated_footpoint_errors']['mae_m']])
    parts+=['<h2>Visible-person detection</h2><p>Same 450 game-2 images, 960 px inference, confidence 0.15, Hungarian IoU≥0.5. Footpoint errors use an oracle camera fitted from labels; they exclude calibration and identity error. Higher recall with lower precision is a trade-off.</p>',table(['Model','Frames','GT people','Matched','Precision','Recall','Oracle footpoint mean (m)'],rows)]
    pitch=read('pitch_lines/results.json')
    if pitch:
        p=pitch['pitch_test'];parts+=['<h2>Independent marking-based camera</h2><p>Predicted image lines plus fixed pitch geometry. Labelled feet are used only for scoring. Unsupported frames are abstentions; large remaining errors prevent deployment.</p>',table(['Frames','Supported frames','People scored','Mean (m)','Median (m)','p90 (m)'],[[p['frames'],p['frames_with_calibration'],p['players'],round(p['mean_m'],2),round(p['median_m'],2),round(p['p90_m'],2)]])]
    for part in ('val','test'):
        tracking=read(f'tracking/{part}.json')
        if tracking:
            rows=[[k,v['gt'],v['matched'],f"{v['precision']:.1%}",f"{v['recall']:.1%}",v['id_switches'],v['matched_transitions']] for k,v in tracking['results'].items()]
            parts += [f'<h2>Tracking motion diagnostic: {part}</h2><p>Identical cached raw detections, 5 Hz. Switches are changes between matched sightings, including gaps; this is not full HOTA/IDF1.</p>',table(['Method','GT people','Matched','Precision','Recall','Switches','Matched transitions'],rows)]
    arsenal=read('arsenal_paired_consistency.json')
    if arsenal:
        rows=[]
        for name,r in arsenal.items():
            p=r['proxy']['reveals'];j=r['hidden_jump_stats']
            rows.append([name,p['n'],p['mean_error_m'],p['median_error_m'],j['over_10_m_per_s'],j['steps'],j['max_m_per_s']])
        parts += ['<h2>Arsenal reappearance consistency</h2><p>Identical corrected detections, camera and role assignments; only hidden-model weights differ. Reference sightings are unverified CV roles, not ground truth. The candidate worsened this check and is not promoted. Jumps include reappearances into preserved observations.</p>',table(['Model','References','Mean proxy error (m)','Median (m)','Steps over 10 m/s','Hidden-related transitions','Maximum (m/s)'],rows)]
    jersey=read('jersey/results.json')
    if jersey:
        parts += ['<h2>Jersey identity candidate</h2><p>Whole-game splits. Number recognition does not verify named identities until mapped to an official lineup; front views without readable digits remain a limitation.</p>']
        validation=jersey.get('validation',{})
        rows=[[threshold,v['decided_tracks'],v['correct_tracks'],f"{v['track_precision']:.1%}",f"{v['track_coverage']:.1%}"] for threshold,v in validation.items()]
        parts.append(table(['Validation vote threshold','Decided tracks','Correct tracks','Precision','Coverage'],rows))
        # Show scalar fields only, never a JSON dump.
        rows=[]
        for group,values in jersey.items():
            if isinstance(values,dict):
                rows.extend([group,k,round(v,3) if isinstance(v,float) else v] for k,v in values.items() if isinstance(v,(str,int,float)))
            elif isinstance(values,(str,int,float)):rows.append(['summary',group,values])
        parts.append(table(['Section','Measure','Value'],rows))
    body=''.join(parts)
    path=root/'review.html';path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Football training review</title><style>body{font:15px system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#17212b}p{line-height:1.6}table{border-collapse:collapse;width:100%;margin:16px 0 28px}th,td{border:1px solid #d5d9de;padding:9px;text-align:left}th{background:#edf0f3}h2{margin-top:36px}td{font-variant-numeric:tabular-nums}</style>'+body+'</html>');print(path)


if __name__=='__main__':
    generate(Path('cv/out/training_20261011'))
