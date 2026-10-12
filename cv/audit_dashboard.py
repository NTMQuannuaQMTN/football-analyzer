"""Readable local tables/figures from executed eight-hour audit artifacts."""
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path('cv/out/audit8_20261011/mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cv.session_report import table, display_rows


ROOT=Path('cv/out/audit8_20261011')


def read(name):
    path=ROOT/name
    return json.loads(path.read_text()) if path.is_file() else None


def generate():
    body=['<h1>Football reconstruction: local audit review</h1><p>Existing assets, held-out evaluation and private candidates. Production exports are preserved. Visible image tracking, pitch placement and hidden-player reconstruction are different tests. A low error on supported measurements does not mean all players were located.</p>']
    links=[('manual_review/review.html','Private Arsenal annotation'),('visible_failures/review.html','Scored visible failures'),
           ('broadcast/namespace_comparison.html','Camera-cut identity comparison'),('active_matches/review.html','19 active export checks'),
           ('broadcast_cameras/review.html','Private Arsenal camera comparison'),
           ('broadcast_coverage/review.html','Missing Arsenal observations'),
           ('broadcast/anonymous_review.mp4','Private anonymous-track movie'),
           ('broadcast/robust_role_review.mp4','Private role-estimate movie')]
    links += [('anonymous_continuity/review.html','Camera-origin teleport diagnosis'),
              ('camera_discontinuities/review.html','Ten remaining jump cases'),
              ('camera_continuity/review.html','Private temporal-camera comparison'),
              ('camera_boundary_review.html','Shot-boundary sample guard'),
              ('hidden_visibility/review.html','Outside-camera versus missed observations'),
              ('camera_foreground_arsenal/review.html','Foreground camera: full paired comparison'),
              ('camera_foreground_arsenal/jumps/review.html','Foreground camera: selected jump images'),
              ('hidden_multimatch/review.html','Seven-match hidden-input stress tests'),
              ('dfl_input_audit/review.html','Raw tracking timing and sampling checks'),
              ('tactical_sensitivity/clean_control/review.html','Tactical geometry: clean control'),
              ('tactical_sensitivity/robust/review.html','Tactical geometry: noise-trained candidate'),
              ('tactical_windows/clean_control/review.html','Retrospective geometry-change errors'),
              ('dense_foreground_full_validation/val.html','Forty-clip camera gate: failed'),
              ('visible_anchor_camera/val.html','Visible-foot anchor check: failed'),
              ('camera_foreground_arsenal/remaining_jumps/review.html','Candidate remaining broadcast failures'),
              ('jersey_inputs/review.html','Private jersey readability and support audit'),
              ('ball_context/review.html','Private ball detection and filled-context audit'),
              ('ball_conditioning/review.html','Stale ball / unknown possession stress'),
              ('camera_foreground_arsenal/unconfirmed_review.mp4','Private anonymous unconfirmed-detection movie'),
              ('api_final_review.html','Actual API checks across twenty parts')]
    links += [('appearance_full_validation/review.html','Forty-clip frozen tracking gates'),
              ('camera_coverage_tradeoffs/review.html','Camera new/common/lost coverage'),
              ('context_fixed_budget/paired_review.html','Equal-update context comparison'),
              ('context_fixed_budget/evaluation_robust/review.html','Context-trained candidate errors'),
              ('context_crossvalidation/review.html','Fresh seven-fold context comparisons'),
              ('api_profile_final_review.html','Actual profile/time-mode API checks'),
              ('context_tactics/review.html','Context error propagated to tactical geometry')]
    body+=['<p><a href="manual_review/review.html">Start the private 18-frame annotation</a> · '
           '<a href="camera_foreground_arsenal/remaining_jumps/review.html">Review remaining broadcast jumps</a></p>',
           '<details><summary>Explore diagnostic reports and private videos</summary><nav>'
           +''.join(f'<a href="{p}">{label}</a> · ' for p,label in links if (ROOT/p).is_file())+'</nav></details>']
    nested=read('context_crossvalidation/review_results.json')
    if nested:
        body+=['<h2>Latest context training: gains and regressions</h2><p>Seven completed nested-match '
               'folds, fresh V3 models, thirty full epochs per control/candidate fit. Five training '
               'matches, separate validation and excluded test; exact visible positions/identities '
               'and simulated camera masks. Means give each previously reviewed German match equal '
               'weight. Stale-context error improves, clean-context error regresses. No checkpoint '
               'is promoted and EPL accuracy remains unverified.</p>',
               table(['Mode','Context','Paired matches','Hidden samples','Control mean m',
                      'Candidate mean m','Change m'],display_rows(nested['macro_rows']))]
    jersey=read('jersey_continuation/results.json')
    if jersey:
        rows=[[threshold,r['crop_count'],r['decided_tracks'],r['correct_tracks'],
               f"{r['track_precision']:.2%}",f"{r['track_coverage']:.2%}"]
              for threshold,r in jersey['validation'].items()]
        body+=['<h2>Jersey continuation: failed validation</h2><p>Nine complete epochs with unchanged '
               'architecture; best epoch one. No fixed threshold meets 90% validation precision on '
               'at least twenty tracks. Held-out test is not run and no jersey model is promoted. '
               'Number voting is retrospective and does not establish named player identity.</p>',
               table(['Validation threshold','Crops','Decided tracks','Correct tracks','Precision','Coverage'],rows)]
    ball=read('ball_context/results.json')
    if ball:
        r=ball['summary'];rows=[[k.replace('_',' ').capitalize(),v] for k,v in r.items() if not isinstance(v,dict)]
        body+=['<h2>Ball context is often estimated</h2><p>Private cached 5 Hz stock detections, not '
               'ground-truth ball labels. Only 606 of 1,579 main-camera samples have a candidate score '
               'at least .2; the tracker can also accept lower-score candidates. The original archive '
               'contains a finite ball in all 601 frames with no per-frame source label. Finite does '
               'not mean observed. No event or export is changed.</p>',table(['Diagnostic','Count / value'],rows),
               table(['Pre-touch 1 Hz context source','Seconds'],r['export_context_sources'].items())]
    stress=read('ball_conditioning/results.json')
    if stress:
        rows=[]
        for mode,record in stress['results'].items():
            for condition,r in record['conditions'].items():
                e=r['model'];rows.append([mode,condition,e['n'],e['mae_m'],round(e['median_m'],3),
                    round(e['p90_m'],3),round(r['paired_delta_mean_m'],3),f"{r['radius_80_coverage']:.2%}"])
        body+=['<h2>Ball and possession sensitivity</h2><p>One held-out German match, two halves, '
               'exact visible positions/IDs/roles and simulated masks. Same 97,116 hidden samples '
               'per mode/condition; no training or selection on these results. Live ball holds use '
               'only past/current samples. This stress is not the measured Arsenal error distribution.</p>',
               table(['Mode','Context','Hidden samples','Mean m','Median m','P90 m','Paired mean change m',
                      'Nominal 80% radius coverage'],rows)]
    rows=[]
    for label,path in [('Eight validation clips','dense_foreground_repro/val.json'),
                       ('Forty validation clips','dense_foreground_full_validation/val.json'),
                       ('Eighteen held-out clips','dense_foreground_repro/test.json')]:
        result=read(path)
        if result:
            for name,r in result['summary'].items():
                e=r['errors']['all'];rows.append([label,name,r['total_frames'],r['supported_frames'],
                    r['total_labelled_people'],e['n'],round(e['mean_m'],3),round(e['p90_m'],3),
                    f"{r['within_5m_recall_all_people']:.2%}"])
    if rows:
        body+=['<h2>Latest camera decision: no promotion</h2><p>Foreground masking initially passes an eight-clip '
               'validation check and improves the held-out game, but the expanded forty-clip validation fails '
               'the unchanged quality tolerances. Parameters were not relaxed. These camera-only results '
               'use labelled visible feet as scoring inputs after image-only camera estimation; detection '
               'and named-player errors are excluded. Different samples have different denominators.</p>',
               table(['Sample','Camera','Frames','Supported frames','All pitch-labelled people','Scored feet',
                      'Mean m','P90 m','Within 5 m / all pitch-labelled people'],rows)]
    result=read('camera_foreground_arsenal/comparison.json')
    if result:
        rows=[]
        for name,r in result['broadcast'].items():
            e=r['paired'];rows.append([name,r['supported_people'],e['n'],round(e['median_m_s'],3),
                round(e['p90_m_s'],3),e['steps_over_10m_s'],round(e['max_m_s'],3)])
        body+=['<h2>Private broadcast: same anonymous transitions</h2><p>Camera-only changes with fixed '
               'cached detections and anonymous IDs. Excessive-speed flags show discontinuity, not accuracy. '
               'The foreground candidate still has a maximum of about 148 m/s and fails broader validation.</p>',
               table(['Camera','Projected people','Common transitions','Median m/s','P90 m/s','Steps >10 m/s',
                      'Maximum m/s'],rows)]
    result=read('appearance_tracking/test.json')
    if result and len(result['per_clip'])==18:
        rows=[]
        for name,r in result['results'].items():
            rows.append([name,r['gt'],r['pred'],f"{r['precision']:.2%}",f"{r['recall']:.2%}",f"{r['identity_f1']:.2%}",f"{r['switches']} / {r['transitions']}"])
        body+=['<h2>Anonymous visible tracking</h2><p>Same stock detections on 18 clips, 2,700 frames, one held-out game. Current/past crops enter appearance association; future trajectory labels score only. No named shirt/lineup accuracy claim.</p>',
               table(['Tracker','GT observations','Predictions','Precision','Recall','Anonymous identity F1','ID switches / transitions'],rows)]
    variants=['legacy/learned_static/kalman','selected/learned_static/kalman','selected/learned_propagated/kalman','raw_detection/learned_propagated/raw']
    rows=[];plot=[]
    for part in ['val','test']:
        result=read(f'visible_pipeline/{part}.json')
        if not result:continue
        for variant in variants:
            r=result['results'][variant];e=r['position_errors_on_matched_supported_people']
            rows.append([part,variant,r['gt_total'],e['n'],round(e['mean_m'],3),round(e['median_m'],3),round(e['p90_m'],3),f"{r['within_5m_recall_all_labelled_people']:.2%}"])
        r=result['results']['selected/learned_propagated/kalman']
        plot.append((part,r))
    body+=['<h2>Camera and detector placement together</h2><p>Visible people include players, keepers and referees. IoU matches are chosen before pitch scoring. Conditional metre errors exclude missed/unsupported measurements; within-5 m recall uses all labelled people.</p>',
           table(['Split','Pipeline','All GT people','Supported matches','Mean m','Median m','P90 m','Within-5 m / all GT'],rows)]
    if plot:
        fig,ax=plt.subplots(figsize=(9,3.7));left=np.zeros(len(plot));names=[x[0] for x in plot]
        for label,key,color in [('Supported and within 5 m','good','#218a56'),('Supported but error >5 m','bad','#cf7341'),('Missing / unsupported','missing','#69798a')]:
            values=[]
            for _,r in plot:
                count=r['matched_within_5m'] if key=='good' else r['matched_camera_supported']-r['matched_within_5m'] if key=='bad' else r['gt_total']-r['matched_camera_supported']
                values.append(100*count/r['gt_total'])
            ax.barh(names,values,left=left,label=label,color=color);left+=values
        ax.set_xlim(0,100);ax.set_xlabel('Percent of all labelled visible-person observations');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=1)
        ax.set_title('Selected tracker + propagated camera: coverage remains incomplete');fig.tight_layout();fig.savefig(ROOT/'visible_coverage.png',dpi=160,bbox_inches='tight');plt.close(fig)
        body+=['<img src="visible_coverage.png" alt="Coverage breakdown; missing observations remain explicit">']
    uncertainty=read('visible_pipeline/uncertainty.json')
    if uncertainty:
        r=uncertainty['test'];rows=[]
        for name,v in r['bins'].items():rows.append([name,v['n'],round(v['radius_80_m'],3),f"{v['radius_coverage']:.2%}",f"{v['predicted_probability_within_5m']:.2%}",f"{v['observed_fraction_within_5m']:.2%}"])
        body+=['<h3>Conditional visible-position uncertainty</h3><p>Validation-fitted radii, same held-out supported/matched people. This excludes missing measurements and unverified names. It is not an EPL confidence calibration.</p>',
               table(['Camera evidence','Test N','Nominal-80% radius m','Coverage','Predicted within-5 m','Observed within-5 m'],rows)]
    candidate_root='hidden_robustness_causal_masks' if (ROOT/'hidden_robustness_causal_masks/test_robust/results.json').is_file() else 'hidden_robustness'
    sources=[('Baseline','hidden_sensitivity/results.json'),('Clean control',f'{candidate_root}/test_clean_control/results.json'),('Robust input',f'{candidate_root}/test_robust/results.json')]
    rows=[]
    conditions=['clean_visible','gaussian_visible_mean_2m','camera_shift_5m_x','identity_swap_10pct_frames','measurement_delay_1s','ball_shift_5m_x']
    records={name:read(path) for name,path in sources}
    for mode in ['live','replay']:
        for condition in conditions:
            for name,value in records.items():
                if not value or mode not in value['results']:continue
                r=value['results'][mode]['conditions'][condition];e=r['model']
                rows.append([mode,condition,name,e['n'],e['mae_m'],round(e['median_m'],2),round(e['p90_m'],2),f"{r['radius_80_coverage']:.2%}"])
    body+=['<h2>Hidden-player perception stresses</h2><p>One DFL tracking match, both halves, exact identities/context, fixed simulated visibility. Same 91,165 common hidden player-timestamps across conditions. Live uses past/current measurements; replay uses future visible measurements. Synthetic corruption is not a fitted EPL error distribution. Radius calibration differs from the original baseline; clean-control and robust candidates share calibration conditions.</p>',
           table(['Mode','Visible-input condition','Model','Hidden N','Mean m','Median m','P90 m','Nominal-80% coverage'],rows)]
    if all(records.values()):
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
        labels=['Exact','2 m noise','5 m camera','ID swaps','1 s delay','5 m ball']
        for ax,mode in zip(axes,['live','replay']):
            for index,(name,value) in enumerate(records.items()):
                if mode not in value['results']:continue
                values=[value['results'][mode]['conditions'][c]['model']['mae_m'] for c in conditions]
                ax.bar(np.arange(len(values))+(index-1)*.25,values,.25,label=name)
            ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.set_title(mode.title());ax.set_ylabel('Mean hidden position error (m)');ax.set_ylim(0,11)
        axes[0].legend(fontsize=8);fig.suptitle('Robust training helps random noise; camera bias remains');fig.tight_layout();fig.savefig(ROOT/'hidden_stress.png',dpi=160);plt.close(fig)
        body+=['<img src="hidden_stress.png" alt="Controlled hidden prediction errors, with systematic-camera failures remaining">']
    rows=[]
    for part in ['val','test']:
        result=read(f'paint_refinement/{part}.json')
        if result:
            for name,r in result['summary'].items():
                rows.append([part,name,r['n'],round(r['mean_m'],3),round(r['p90_m'],3),result['accepted_refinements']])
    if rows:
        body+=['<h2>Paint refinement failed to generalise</h2><p>Current-image alignment improved validation but slightly worsened the held-out game. Exactly the same supported feet are compared. Better paint-distance residuals do not guarantee more accurate pitch positions. The method remains experimental.</p>',
               table(['Split','Camera','Paired feet','Mean m','P90 m','Accepted refinements'],rows)]
    result=read('hidden_visibility/results.json')
    if result:
        rows=[]
        for name,model in result['models'].items():
            for group,r in model['groups'].items():
                e=r['error'];inside=r['predicted_inside_camera_fraction']
                rows.append([name,group,e['n'],e['mae_m'],round(e['p90_m'],2),f"{inside:.2%}" if inside is not None else 'N/A',f"{r['radius_80_coverage']:.2%}"])
        body+=['<h2>Off-camera versus missed detections</h2><p>Controlled DFL evaluation with transferred camera polygons. 47,181 unavailable player-timestamps are truly outside a known footprint, 809 are inside but not observed, and 49,126 have no usable camera footprint. These labels score only. Missing observations must not automatically be forced outside the camera.</p>',
               table(['Mode / model','Reference group','Hidden N','Mean m','P90 m','Prediction inside camera','Nominal-80% coverage'],rows)]
    result=read('ground_tracking/val.json')
    if result:
        rows=[[name,r['gt'],f"{r['precision']:.2%}",f"{r['recall']:.2%}",f"{r['identity_f1']:.2%}",r['switches']] for name,r in result['results'].items()]
        body+=['<h2>Alternative tracking motion: validation rejection</h2><p>Identical cached detections, eight clips, 1,200 validation frames. Ground-paint affine compensation and no-motion tracking failed the predeclared validation gate, so neither was tested or promoted.</p>',
               table(['Motion','Reference people','Precision','Recall','Anonymous identity F1','Switches'],rows)]
    history=read('physical_camera/history.json');finished=read('physical_camera/finished.json')
    if history:
        validation=[x for x in history if 'validation' in x]
        best=min(validation,key=lambda x:x['validation']['selection_cost_m']);v=best['validation']
        body+=['<h2>Physical-camera experiment</h2><p>'+('Training finished.' if finished else 'Training is still running; this is a validation progress snapshot, not an accepted result.')+' Image only at inference. Best checkpoint is selected on a fixed 250-frame validation subset; full validation and its gate are separate.</p>',
               table(['Epochs seen','Best epoch','Subset frames','Accepted people','Conditional mean m','P90 m','Selection cost m'],[[len(history),best['epoch'],v['total_frames'],v['accepted_people'],round(v['accepted_mean_m'],3) if v['accepted_mean_m'] is not None else None,round(v['accepted_p90_m'],3) if v['accepted_p90_m'] is not None else None,round(v['selection_cost_m'],3)]])]
    rows=[]
    for experiment in ['physical_camera','physical_world_weight']:
        result=read(f'{experiment}/evaluation/val.json')
        quality=read(f'{experiment}/evaluation/val_quality.json')
        if not result:continue
        for name,r in result['summary'].items():
            rows.append([experiment,r['frames'],r['supported_frames'],r['people'],round(r['mean_m'],3),round(r['median_m'],3),round(r['p90_m'],3),'passed' if quality and quality['selected'] is not None else 'failed' if quality else 'pending'])
    if rows:
        body+=['<h3>Full physical-camera validation</h3><p>Image-only inference, validation-selected checkpoint. The frozen operating gate requires mean ≤3 m, p90 ≤5 m and frame coverage ≥40%. Failed candidates are not tested or promoted. A stronger-loss continuation is separate from the first run.</p>',table(['Experiment','Validation frames','Supported frames','Labelled feet','Mean m','Median m','P90 m','Quality gate'],rows)]
    playback_rows=[]
    for label,path in [('Original 1x','browser/playback.json'),
                       ('Selected player 1x','browser/playback_selected_final.json'),
                       ('Selected player 8x','browser/playback_selected_8x_final.json'),
                       ('Selected player 32x','browser/playback_selected_32x_final.json')]:
        playback=read(path)
        if playback:
            playback_rows.append([label,round(playback['duration_s'],3),round(playback['raf_rate_hz'],2),
                round(playback['position_updates_hz'],2),round(playback['median_interval_ms'],2),
                round(playback['p90_interval_ms'],2),len(playback['longTasks'])])
    if playback_rows:
        body+=['<h2>Actual browser presentation checks</h2><p>Isolated headless Chrome, local '
               'development studio and preserved Arsenal archive. Presentation throughput is not '
               'source sampling, positional accuracy or the user’s physical display performance. '
               'The selected panel makes API requests when the integer playback second changes.</p>',
               table(['Condition','Duration s','Animation Hz','Position updates Hz','Median interval ms',
                      'P90 interval ms','Long tasks'],playback_rows)]
    body+=['<h2>Remaining limits</h2><p>No independent hidden EPL truth, no verified Arsenal shirt mapping, incomplete visible placement coverage, systematic camera error and measured-position jumps. Private candidate improvement is not production promotion. See the Markdown audit and user-action checklist for exact commands, source changes and review needs.</p>']
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Football reconstruction audit</title><style>body{font:15px system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#f8fafc;color:#17212b}p{line-height:1.6}table{border-collapse:collapse;width:100%;margin:16px 0 30px;font-variant-numeric:tabular-nums}th,td{border:1px solid #cbd5e1;padding:8px;text-align:left}th{background:#e2e8f0}h2{margin-top:36px}img{max-width:100%;display:block;margin:20px auto}a{color:#1d4ed8}</style>'+''.join(body)+'</html>'
    (ROOT/'review.html').write_text(page)
    (ROOT/'dashboard_sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.json') if p.name!='dashboard_sources.json'},indent=2))
    print(ROOT/'review.html',flush=True)


if __name__=='__main__':generate()
