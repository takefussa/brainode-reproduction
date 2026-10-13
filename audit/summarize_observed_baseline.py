"""Verify and report the fixed-final-epoch observed BrainODE baseline."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from evaluation import shape_errors
from lab_shapes import ShapePCA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report=PROJECT/'artifacts/training_runs/adni1_observed_v1'
    config=json.loads((report/'config.json').read_text())
    progress=json.loads((report/'progress.json').read_text())
    output,data_root=Path(config['output']),Path(config['data_root'])
    if progress['status']!='completed_observed_baseline' or progress['epochs_completed']!=100:
        raise ValueError('completed fixed-final-epoch baseline required')
    if digest(output/'latest.pt')!=progress['checkpoint_sha256'] or digest(output/'future_predictions.npz')!=progress['evaluation']['prediction_file_sha256']:
        raise ValueError('completed artifacts changed')
    for name,expected in config['data_sha256'].items():
        if digest(data_root/name)!=expected:
            raise ValueError('frozen data changed')
    with np.load(data_root/'observations.npz',allow_pickle=False) as file:
        data={k:file[k].copy() for k in file.files}
    with np.load(output/'future_predictions.npz',allow_pickle=False) as file:
        predictions={k:file[k].copy() for k in file.files}
    np.testing.assert_array_equal(data['faces'],predictions['faces'])
    pca=ShapePCA.load(data_root/'pca.npz')
    with (report/'future_metrics.csv').open() as file:
        rows=list(csv.DictReader(file))
    with (report/'epochs.csv').open() as file:
        history=list(csv.DictReader(file))
    if len(rows)!=175 or len(history)!=100 or any(int(r['ordered_pairs'])!=3412 or int(r['optimizer_steps'])!=150 for r in history):
        raise ValueError('training or evaluation coverage mismatch')
    indices={str(uid):i for i,uid in enumerate(data['uids'])}
    groups={};areas=[]
    for index,row in enumerate(rows):
        source,target=indices[row['source_uid']],indices[row['target_uid']]
        for i in (source,target):
            if str(data['pids'][i])!=row['pid'] or str(data['splits'][i])!='test':
                raise ValueError('evaluation patient/split mismatch')
        if str(predictions['pids'][index])!=row['pid'] or str(predictions['target_uids'][index])!=row['target_uid']:
            raise ValueError('prediction ordering differs')
        predicted=predictions['vertices'][index]
        actual=shape_errors(predicted,data['vertices'][target])['mean_vertex_euclidean']
        if abs(actual-float(row['prediction_mean_vertex_euclidean']))>1e-12:
            raise ValueError('saved score differs from predicted mesh')
        smooth=pca.inverse_transform(data['coefficients'][source:source+1]).reshape(-1,3)
        row['pca_static_mean_vertex_euclidean']=shape_errors(smooth,data['vertices'][target])['mean_vertex_euclidean']
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(predicted),o3d.utility.Vector3iVector(data['faces']))
        pairs=np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1,2)
        if len(pairs)!=int(row['prediction_self_intersection_pairs']):
            raise ValueError('saved surface intersections differ')
        ids=np.unique(pairs);triangles=predicted[data['faces']]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        fraction=float(100*area[ids].sum()/area.sum()) if len(ids) else 0.
        row['prediction_intersecting_surface_area_percent']=fraction
        if len(ids):areas.append(fraction)
        groups.setdefault(row['pid'],[]).append(index)
    keys=['prediction_mean_vertex_euclidean','static_source_mean_vertex_euclidean','pca_static_mean_vertex_euclidean']
    patient_means={pid:np.array([[float(rows[i][k]) for k in keys] for i in group]).mean(0) for pid,group in groups.items()}
    mean=np.stack(list(patient_means.values())).mean(0)
    summary=dict(completed_jst='2026-10-07 22:22',epochs=100,elapsed_minutes=progress['elapsed_seconds']/60,
        test_subjects=len(groups),test_future_targets=len(rows),
        subject_macro_mean_vertex_error=dict(zip(keys,mean.tolist())),
        relative_error_reduction_vs_raw_static=float(1-mean[0]/mean[1]),
        relative_error_reduction_vs_pca_static=float(1-mean[0]/mean[2]),
        patients_improved_vs_raw_static=sum(v[0]<v[1] for v in patient_means.values()),
        patients_improved_vs_pca_static=sum(v[0]<v[2] for v in patient_means.values()),
        predicted_meshes_with_self_intersection=len(areas),
        intersecting_surface_area_percent=dict(min=min(areas),median=float(np.median(areas)),max=max(areas)),
        source_data_and_completed_artifacts_modified=False,cnn_or_algorithm2_used=False,
        units='assumed mm: one numerical source coordinate unit interpreted as one mm',
        limitations=['observed-only baseline; continuous-c CNN and Algorithm 2 remain untrained',
                     'ADNI1 test has no CN-to-AD subjects',
                     'shared face order and small vertex error do not prove anatomical correspondence or valid surfaces'])
    # Display the patient at the middle rank of mean prediction error, and their
    # latest target. This is a transparent display rule, not checkpoint selection.
    ranked=sorted(patient_means,key=lambda p:(patient_means[p][0],p))
    pid=ranked[len(ranked)//2]
    index=max(groups[pid],key=lambda i:(rows[i]['target_day'],int(rows[i]['target_uid'])))
    row=rows[index];source,target=indices[row['source_uid']],indices[row['target_uid']]
    summary['preview']=dict(pid=pid,source_uid=row['source_uid'],target_uid=row['target_uid'],
        selection='middle-rank test patient by mean vertex prediction error, latest target of that patient',
        prediction_error=float(row['prediction_mean_vertex_euclidean']),
        static_error=float(row['static_source_mean_vertex_euclidean']),
        intersections=int(row['prediction_self_intersection_pairs']))
    (report/'RESULTS.json').write_text(json.dumps(summary,indent=2,default=lambda v:int(v))+'\n')
    with (report/'RESULTS.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot([int(r['epoch']) for r in history],[float(r['train_patient_mean_squared_pca_l2']) for r in history])
    axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Training patient mean squared PCA L2');axes[0].grid(alpha=.2)
    values=np.stack(list(patient_means.values()))
    axes[1].scatter(values[:,1],values[:,0],s=18,alpha=.75,label='Test patients')
    limit=max(values[:,:2].max()*1.05,.1);axes[1].plot([0,limit],[0,limit],color='gray',lw=1)
    axes[1].set(xlim=(0,limit),ylim=(0,limit),xlabel='Static initial mesh error (assumed mm)',ylabel='BrainODE error (assumed mm)')
    axes[1].grid(alpha=.2);axes[1].legend()
    fig.savefig(report/'results.png',dpi=170);fig.savefig(report/'results.pdf');plt.close(fig)
    shapes=[data['vertices'][source],data['vertices'][target],predictions['vertices'][index]]
    titles=['Initial observation: '+row['source_day'],'Future observation: '+row['target_day'],'BrainODE prediction: '+row['target_day']]
    viewer=make_subplots(rows=1,cols=3,specs=[[{'type':'scene'}]*3],subplot_titles=titles)
    faces=data['faces']
    for column,(v,color) in enumerate(zip(shapes,['#87939b','#2697ce','#e8a838']),1):
        viewer.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=color,name=titles[column-1]),row=1,col=column)
    union=np.concatenate(shapes);lower,upper=union.min(0)-1,union.max(0)+1
    scene=dict(aspectmode='data',xaxis=dict(range=[lower[0],upper[0]]),yaxis=dict(range=[lower[1],upper[1]]),zaxis=dict(range=[lower[2],upper[2]]))
    viewer.update_scenes(scene)
    viewer.update_layout(title=f"{pid}: observed-only final-epoch forecast, error {summary['preview']['prediction_error']:.3f} assumed mm; self-intersection pairs {summary['preview']['intersections']}",height=650)
    viewer.write_html(report/'prediction_preview.html',include_plotlyjs=True,full_html=True)
    lines=['# ADNI1 初期BrainODEの結果','',
        '観測だけの100 epochsを2026-10-07 22:22 JSTに完了した。学習と最終評価の実行時間は約92.3分。train299患者・3,412時間対、test60患者・175将来撮像。未来診断とtestを学習・checkpoint選択に使用していない。','',
        '| 比較 | 患者ごとの平均頂点距離を等重みで平均（仮定mm） |','|---|---:|',
        f'| BrainODE | {mean[0]:.4f} |',f'| 初期の元meshを維持 | {mean[1]:.4f} |',f'| 初期のPCA再構成meshを維持 | {mean[2]:.4f} |','',
        f"元meshの維持に比べ平均誤差は{summary['relative_error_reduction_vs_raw_static']*100:.2f}%減少し、60患者中{summary['patients_improved_vs_raw_static']}患者で改善した。PCA再構成meshの維持との比較は{summary['relative_error_reduction_vs_pca_static']*100:.2f}%減少、{summary['patients_improved_vs_pca_static']}/60患者で改善。",'',
        f"予測175形状中{len(areas)}形状に自己交差がある。関係する面積割合は{summary['intersecting_surface_area_percent']}%。予測座標の誤差改善だけで表面品質の問題が解決したとは扱わない。",'',
        '距離単位は元headerの数値1を1mmとする既存の仮定。PCA targetだけでなく元観測meshを主targetにした結果。連続cのCNNとAlgorithm 2は未学習で、ADNI1 testにはCN→AD患者がいない。','',
        '[全数値](RESULTS.json)、[撮像ごとの比較](RESULTS.csv)、[lossと患者ごとの比較](results.png)、[予測と観測の回転可能3D](prediction_preview.html)。','',
        f"3D表示例は予測誤差の患者順位の中央にある{pid}の最後のtargetを機械的に選んだ。初期観測・将来観測・予測は共通canonical mesh座標にあり、native MRIへ登録した表示ではない。",'',
        '次段階はtrain端点からの式(4)のpseudo連続c生成、voxel入力3D CNN、追加ODEとAlgorithm 2評価。未記載の設定を具体化して学習前に提示する。','']
    (report/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2,default=lambda v:int(v)))


if __name__=='__main__':
    main()
