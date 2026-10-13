"""Measured/predicted LV review export with explicit source-native RAS mapping."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from export_helpers import write_mesh, voxel_boundary
from pipeline import SIDES, LABELS, paths, digest, read_rows, write_json, ordered

PAPER='https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf'


def transforms(path):
    with np.load(path,allow_pickle=False) as a:
        matrix=a['icp_matrix'].copy();center=a['centroid_header_world'].copy()
    canonical_to_native=np.linalg.inv(matrix)
    canonical_to_native[:3,3]+=center
    native_to_canonical=matrix.copy()
    native_to_canonical[:3,3]-=matrix[:3,:3]@center
    np.testing.assert_allclose(canonical_to_native@native_to_canonical,np.eye(4),atol=1e-8)
    return canonical_to_native,native_to_canonical


def numeric(rows):
    output=[]
    for row in rows:
        value=dict(row)
        for key in ('prediction_mean_vertex_euclidean','static_source_mean_vertex_euclidean','horizon_years','target_time'):
            if key in value: value[key]=float(value[key])
        output.append(value)
    return output


def macro(rows,key):
    groups={}
    for r in rows: groups.setdefault(r['pid'],[]).append(float(r[key]))
    return float(np.mean([np.mean(v) for v in groups.values()]))


def latest(rows):
    end={pid:max(r['target_time'] for r in rows if r['pid']==pid) for pid in {r['pid'] for r in rows}}
    return [r for r in rows if r['target_time']==end[r['pid']]]


def compare(c,side):
    p=paths(c,side)
    initial=numeric(read_rows(p['report']/'observed/future_metrics.csv'))
    final=numeric(read_rows(p['report']/'continuous/evaluate/future_metrics.csv'))
    lookup={(r['pid'],r['source_uid'],r['target_uid']):r for r in final}
    for r in initial: r['target_time']=lookup[(r['pid'],r['source_uid'],r['target_uid'])]['target_time']
    if len(initial)!=len(final) or len(final)!=c['test_future_targets']: raise ValueError('comparison coverage mismatch')
    result={}
    for name,a,b in [('all',initial,final),('latest',latest(initial),latest(final))]:
        base=macro(a,'prediction_mean_vertex_euclidean');pred=macro(b,'prediction_mean_vertex_euclidean')
        static=macro(b,'static_source_mean_vertex_euclidean')
        patient_ids=sorted({r['pid'] for r in b})
        difference=np.array([np.mean([r['prediction_mean_vertex_euclidean'] for r in b if r['pid']==pid])-
            np.mean([r['prediction_mean_vertex_euclidean'] for r in a if r['pid']==pid]) for pid in patient_ids])
        rng=np.random.default_rng(42)
        bootstrap=difference[rng.integers(0,len(patient_ids),size=(10000,len(patient_ids)))].mean(1)
        result[name]=dict(subjects=len(patient_ids),targets=len(b),initial_mean_vertex_distance_mm=base,
            final_algorithm2_mean_vertex_distance_mm=pred,static_mean_vertex_distance_mm=static,
            final_improvement_over_initial_percent=100*(base-pred)/base,
            final_improvement_over_static_percent=100*(static-pred)/static,
            paired_patient_final_minus_initial_mm=float(difference.mean()),
            paired_patient_bootstrap_95_percent_interval_mm=np.quantile(bootstrap,[.025,.975]).tolist())
        if 'initial_N10_fixed_c_euclidean' in b[0]:
            old = macro(b,'initial_N10_fixed_c_euclidean')
            updated = macro(b,'augmented_N10_fixed_c_euclidean')
            result[name]['matched_N10_controls'] = dict(initial_fixed_c_mm=old,augmented_fixed_c_mm=updated,
                final_progressive_c_mm=pred,training_change_mm=updated-old,progressive_c_change_mm=pred-updated,
                interpretation='Same ten RK4 intervals. Training change combines extra optimization and pseudo samples; no pseudo-only causal claim.')
    result.update(region=side,primary_metric='corresponding-vertex Euclidean distance; patient macro mean; all retained targets',
        paper_reference=dict(LV_irregular_1shot_mm=1.673,LV_irregular_4shot_mm=1.959,url=PAPER,
            direct_comparison_valid=False,reason='different dataset mix, vertex/template processing, split, assumed units and unpublished CNN settings'),
        continuous_c=json.loads((p['report']/'continuous/cnn/evaluation.json').read_text()),
        geometry=json.loads((p['report']/'continuous/evaluate/evaluation.json').read_text()),
        settings_differences=['ADNI1 only','100 initial + 20 supplementary ODE epochs',
            'PCA uses all train scans','LV-Net template with BrainODE A.1 seven losses',
            'train-only fixed physical 128^3 grid and supplementary simple CNN','Algorithm 2 N=10'],
        supervision_limit='Two train conversion patients, four pairs; no held-out converters; c is not clinically validated')
    result['cnn_batch_execution']=json.loads((p['report']/'continuous_config.json').read_text())['cnn'].get('gradient_accumulation','single effective batch')
    return result


def export_all(c):
    from mesh_cohort import subject_folder
    root=Path(c['bundle']);root.mkdir(parents=True,exist_ok=True)
    data_dir=root/'data';data_dir.mkdir(exist_ok=True)
    summaries={side:compare(c,side) for side in SIDES}
    write_json(root/'scores.json',summaries)
    table={r['uid']:r for r in read_rows(c['table'])};source_uid=c['review_source_uid'];target_uid=c['review_target_uid']
    source_row=table[source_uid];target_row=table[target_uid]
    if source_row['pid']!=target_row['pid'] or source_row['split']!='test': raise ValueError('review case must be same held-out patient')
    source_image=nib.load(source_row['parcellation_path']);target_image=nib.load(target_row['parcellation_path'])
    source_array=np.asarray(source_image.dataobj);target_array=np.asarray(target_image.dataobj)
    model_records=[];volume_records=[];case=dict(pid=source_row['pid'],source_uid=source_uid,target_uid=target_uid,
        source_day=source_row['day'],target_day=target_row['day'],horizon_years=float(target_row['age'])-float(source_row['age']),
        units='assumed mm; original header units unspecified',coordinate_frame='source scan native RAS',
        observed_future_is_prediction=False,source_images_sha256={r['parcellation_path']:digest(r['parcellation_path']) for r in (source_row,target_row)})
    arrays={};colors={'segmentation':'#559ade','observed_mesh':'#fdae61','initial_prediction':'#ab74c4','final_prediction':'#dd4545'}
    for side in SIDES:
        p=paths(c,side);directory=p['mesh']/subject_folder(source_row['pid'])
        source_mesh=directory/(source_uid+'.npz');target_mesh=directory/(target_uid+'.npz')
        if not c.get('FUNCTIONAL_TEST_ONLY'):
            manifest=json.loads((directory/'manifest.json').read_text())
            for uid,file in [(source_uid,source_mesh),(target_uid,target_mesh)]:
                record=next(r for r in manifest['scans'] if r['uid']==uid)
                if digest(file)!=record['artifact_sha256'][uid+'.npz']:
                    raise ValueError('review mesh changed: '+str(file))
        source_c2n,_=transforms(source_mesh);_,target_n2c=transforms(target_mesh)
        for date,image,raw,affine in [('baseline',source_image,source_array,source_image.affine),
                                     ('future_observed',target_image,target_array,source_c2n@target_n2c@target_image.affine)]:
            mask=np.isin(raw,LABELS[side]);file=data_dir/f'{date}_{side}_LV_mask.nii.gz'
            header=image.header.copy();header.set_data_dtype(np.uint8);header.set_xyzt_units('mm')
            out=nib.Nifti1Image(mask.astype(np.uint8),affine,header=header)
            out.set_qform(affine,code=1);out.set_sform(affine,code=1);nib.save(out,file)
            reread=nib.load(file)
            np.testing.assert_array_equal(np.asarray(reread.dataobj),mask.astype(np.uint8))
            np.testing.assert_allclose(reread.affine,affine,atol=2e-5)
            v,f=voxel_boundary(mask,affine);prefix=data_dir/f'{date}_{side}_LV_segmentation_surface'
            write_mesh(prefix,v,f);key=f'{date}_{side}_segmentation';arrays[key]=(v,f)
            model_records.append(dict(key=key,file=str(prefix.with_suffix('.vtk').relative_to(root)),date=date,side=side,
                kind='segmentation',label=f'{side}: {date} segmentation',color=colors['segmentation']))
            volume_records.append(dict(file=str(file.relative_to(root)),date=date,side=side,voxels=int(mask.sum()),
                affine=affine.tolist(),labels=LABELS[side],resampled=False))
        for date,file in [('baseline',source_mesh),('future_observed',target_mesh)]:
            with np.load(file,allow_pickle=False) as a: v=a['vertices'].copy();f=a['faces'].copy()
            v=nib.affines.apply_affine(source_c2n,v)
            prefix=data_dir/f'{date}_{side}_LV_fitted_mesh';write_mesh(prefix,v,f)
            key=f'{date}_{side}_observed_mesh';arrays[key]=(v,f)
            model_records.append(dict(key=key,file=str(prefix.with_suffix('.vtk').relative_to(root)),date=date,side=side,
                kind='observed_mesh',label=f'{side}: {date} fitted mesh',color=colors['observed_mesh']))
        for kind,path in [('initial_prediction',p['initial']/'future_predictions.npz'),
                          ('final_prediction',p['continuous']/'evaluate/future_predictions.npz')]:
            with np.load(path,allow_pickle=False) as a:
                matches=np.flatnonzero((a['source_uids']==source_uid)&(a['target_uids']==target_uid))
                if len(matches)!=1: raise ValueError('prediction case lookup failed')
                v=nib.affines.apply_affine(source_c2n,a['vertices'][int(matches[0])]);f=a['faces'].copy()
            prefix=data_dir/f'future_{side}_LV_{kind}';write_mesh(prefix,v,f)
            key=f'future_{side}_{kind}';arrays[key]=(v,f)
            model_records.append(dict(key=key,file=str(prefix.with_suffix('.vtk').relative_to(root)),date='future_predicted',side=side,
                kind=kind,label=f'{side}: {kind}',color=colors[kind]))
        provenance=root/'provenance'/side;provenance.mkdir(parents=True,exist_ok=True)
        for name,path in [('observed_config.json',p['report']/'observed_config.json'),
                          ('continuous_config.json',p['report']/'continuous_config.json'),('grid.json',p['report']/'grid.json'),
                          ('data_summary.json',p['data']/'summary.json'),('source_mesh_manifest.json',directory/'manifest.json'),
                          ('final_metrics.csv',p['report']/'continuous/evaluate/future_metrics.csv'),
                          ('latest_metrics.csv',p['report']/'continuous/evaluate/latest_visit_metrics.csv')]: shutil.copy2(path,provenance/name)
        # All held-out final meshes are also available in the heavy output;
        # the portable case bundle keeps download size manageable.
    case.update(models=model_records,volumes=volume_records)
    write_json(root/'review_manifest.json',case)
    figure=go.Figure()
    for record in model_records:
        v,f=arrays[record['key']]
        visible=(record['date']=='baseline' and record['kind']=='observed_mesh') or record['kind']=='final_prediction'
        figure.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            name=record['label'],color=record['color'],opacity=.35 if record['kind']=='segmentation' else .85,
            visible=True if visible else 'legendonly',showlegend=True))
    figure.update_layout(title=f'ADNI1 {source_row["pid"]}: measured baseline / predicted LV, {case["horizon_years"]:.2f} years',
        scene=dict(aspectmode='data',xaxis_title='R (assumed mm)',yaxis_title='A (assumed mm)',zaxis_title='S (assumed mm)'),
        legend=dict(itemsizing='constant'),height=850)
    figure.write_html(root/'viewer.html',include_plotlyjs=True,full_html=True)
    (root/'load_in_slicer.py').write_text('''"""Run inside 3D Slicer. Models specify SPACE=RAS; masks retain native grids."""
from pathlib import Path
import json
import slicer
ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'review_manifest.json').read_text())
for record in manifest['models']:
    node = slicer.util.loadModel(str(ROOT / record['file']))
    node.SetName(record['label'])
    color = record['color'].lstrip('#')
    node.GetDisplayNode().SetColor(*[int(color[i:i+2],16)/255 for i in (0,2,4)])
    node.GetDisplayNode().SetVisibility(record['kind']=='final_prediction' or (record['date']=='baseline' and record['kind']=='observed_mesh'))
for record in manifest['volumes']:
    node = slicer.util.loadLabelVolume(str(ROOT / record['file']))
    node.SetName(record['date'] + ' ' + record['side'] + ' LV segmentation')
slicer.util.resetThreeDViews()
''')
    lines=['# ADNI1 左右側脳室の設定・評価・3D出力','',
        f"表示例: {source_row['pid']}、{source_row['day']} → {target_row['day']}（{case['horizon_years']:.4f}年）。",'',
        '左右別モデル。全468患者・1,405撮像。train 389患者/1,150撮像、test 79患者/255撮像。CN/AD撮像を保持し、MCI撮像を除外。年齢制限なし。',
        'time=(age−55.7288)/37.2931。既存患者分割を継承し、testから再fitしない。',
        'mesh: 2,490頂点/4,976面、各撮像5,000更新。公開LV-Net templateにBrainODE A.1の7損失を使用。訓練時点ごとのmeshは次の撮像の初期形状。',
        'PCA150は全train撮像のみ。公開ODEは100 epoch、その後Eq.(4) train-only擬似36例、CNN30 epoch、ODE追加20 epoch。Algorithm 2 N=10。最終epoch固定。','',
        '|側|最新時点: 初期ODE (mm)|最終予測 (mm)|静的mesh (mm)|初期ODEからの改善 (%)|',
        '|---|---:|---:|---:|---:|']
    for side in SIDES:
        v=summaries[side]['latest'];lines.append(f"|{side}|{v['initial_mean_vertex_distance_mm']:.6f}|{v['final_algorithm2_mean_vertex_distance_mm']:.6f}|{v['static_mean_vertex_distance_mm']:.6f}|{v['final_improvement_over_initial_percent']:.3f}|")
    lines.extend(['','正の改善率は誤差減少、負は悪化。患者平均を主指標とし、最新同時点のrepeatも保持。全将来時点と患者bootstrap区間はscores.jsonに保存。',
        'plant-ai05ではCNNをmicrobatch4で計算し、effective batch16ごとにAdamWを1回更新。BCEの各例の勾配は1/Nで加算する。層構成、データ順、更新回数、FP32設定は維持。GPU間の数値誤差はあり、元実行とのbitwise同一性は保証しない。',
        f'論文 Table 2 のLV irregular 1-shotは1.673 mm（4-shot 1.959 mm）。データ構成・split・template・単位・非公開CNN設定が異なるため、論文超えの判断には使えない。[論文]({PAPER})。',
        '原画像header単位は未指定。座標値1を1 mmと解釈。自己交差は除外・修復せずgeometry欄に記録。連続cは診断/擬似ラベルから推定した値で、臨床的な連続認知尺度としての検証はない。','',
        'viewer.htmlは通信なしで開ける。凡例でbaseline、将来の実測segmentation、実測fitted mesh、初期ODE、最終予測を切り替える。',
        'Slicer: data/future_left_LV_final_prediction.vtk と future_right_LV_final_prediction.vtk が将来予測。baseline_*_LV_mask.nii.gz が初期segmentation。future_observed_*_LV_mask.nii.gz は将来実測。',
        'VTKはSPACE=RAS、OBJもRAS。maskは256³原gridを保持し、未来maskのaffineだけをrigid変換して初期画像の座標へ配置。resamplingなし。左右それぞれのsource ICPを逆変換する。',
        '一括読込: Slicer Python Consoleで `script="/展開先/load_in_slicer.py"; exec(compile(open(script).read(),script,"exec"),{"__file__":script})`。',
        '書出しファイルは数値/affine/頂点/面のreadbackを検証。Slicer GUIはサーバー上にないため実画面での確認は未実施。',''])
    (root/'README.md').write_text('\n'.join(lines))
    if c.get('FUNCTIONAL_TEST_ONLY'):
        (root/'FUNCTIONAL_TEST_ONLY.txt').write_text('Synthetic coefficient integration test. Not measured forecasts or scientific accuracy.\n')
        (root/'README.md').write_text('FUNCTIONAL TEST ONLY. Synthetic shapes and coefficients. All accuracy numbers in this directory are test outputs.\n\n'+(root/'README.md').read_text())
    hashes={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='checksums.json'}
    write_json(root/'checksums.json',dict(files_sha256=hashes,created_utc=datetime.now(timezone.utc).isoformat()))
    zip_path=Path(c['bundle_zip']);zip_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=zip_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.rglob('*')):
            if p.is_file(): archive.write(p,arcname=str(Path(root.name)/p.relative_to(root)))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP integrity failure')
    temporary.replace(zip_path)
    write_json(Path(c['report'])/'final_results.json',dict(scores=summaries,bundle=str(root),bundle_zip=str(zip_path),zip_sha256=digest(zip_path)))
    return [zip_path,root/'checksums.json',root/'scores.json',Path(c['report'])/'final_results.json']
