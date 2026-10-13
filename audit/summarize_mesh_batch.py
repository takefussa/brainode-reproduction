"""Check saved mesh chains and report all completed and unresolved observations."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mesh_cohort import ordered_scans, subject_folder
from mesh_initialization import initialize_template


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('light audit reports must remain under the home project')
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root/'plan.json').read_text())
    progress = json.loads((root/'progress.json').read_text())
    with Path(plan['table']).open() as file:
        all_rows = list(csv.DictReader(file))
    template = trimesh.load_mesh(PROJECT/'dataset/Shapes/hippocampus_template_mesh.obj', process=False)
    records, remaining = [], []
    for subject in plan['subjects']:
        expected = ordered_scans(r for r in all_rows if r['pid']==subject)
        directory = root/subject_folder(subject)
        manifest = json.loads((directory/'manifest.json').read_text()) if (directory/'manifest.json').exists() else {'scans':[]}
        scans = manifest['scans']
        if scans:
            for name, digest in manifest['code_provenance'].items():
                if hashlib.sha256((Path(plan['code_snapshot'])/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('fitting code differs from frozen snapshot')
            if hashlib.sha256((PROJECT/'dataset/Shapes/hippocampus_template_mesh.obj').read_bytes()).hexdigest()!=manifest['template_sha256']:
                raise ValueError('template differs from the fitted template')
        if len(scans)>len(expected):
            raise ValueError('saved series exceeds planned observations')
        source = None
        for index, record in enumerate(scans):
            row = expected[index]
            if (record['uid'],record['day'],record['age'],record['diagnosis'],record['cohort'],record['split']) != (
                    row['uid'],row['day'],float(row['age']),row['class'],row['cohort'],row['split']):
                raise ValueError('saved observation metadata differs from frozen plan')
            if record['source_uid'] != (expected[index-1]['uid'] if index else 'template'):
                raise ValueError('chronological predecessor differs')
            for name, digest in record['artifact_sha256'].items():
                if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('saved artifact hash differs')
            with np.load(directory/f"{record['uid']}.npz") as data:
                np.testing.assert_array_equal(data['faces'],template.faces)
                if index==0:
                    source,_ = initialize_template(template.vertices,template.faces,mode='mask-volume',
                                                   mask_volume=record['mask_volume_mm3'])
                np.testing.assert_array_equal(data['source_vertices'],source)
                source = data['vertices'].copy()
                if source.shape!=(1490,3) or not np.isfinite(source).all():
                    raise ValueError('invalid saved vertices')
            records.append(dict(subject=subject,uid=record['uid'],cohort=record['cohort'],day=record['day'],
                age=record['age'],diagnosis=record['diagnosis'],split=record['split'],
                mask_volume=record['mask_volume_mm3'],mesh_volume=record['after']['volume_mm3'],
                volume_bias_percent=record['volume_bias_percent'],dice=record['overlap']['dice'],
                boundary_mean_mm=record['after']['boundary_to_surface_mean_mm'],
                intersections=record['after']['self_intersecting_triangle_pairs'],
                qc_flags=';'.join(record['qc_flags']),fitting_seconds=record['elapsed_seconds']))
        remaining.extend(dict(subject=subject,uid=row['uid'],reason=progress['subjects'][subject]) for row in expected[len(scans):])
    if records:
        with (output/'mesh_metrics.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        labels=[f"{r['cohort']}\n{r['uid']}" for r in records]
        x=np.arange(len(records))
        if len(records)<=40:
            axes[0].bar(x,[r['volume_bias_percent'] for r in records],color='#388cb5');axes[0].axhline(0,color='black',lw=.7)
            axes[0].set_ylabel('Volume bias (%)')
            axes[1].bar(x,[r['dice'] or 0 for r in records],color='#388cb5');axes[1].set_ylim(0,1);axes[1].set_ylabel('Dice')
            axes[2].bar(x,[r['intersections'] for r in records],color='#b74a44');axes[2].set_ylabel('Self-intersection pairs')
            for ax in axes:
                ax.set_xticks(x,labels,rotation=90,fontsize=6)
        else:
            axes[0].hist([r['volume_bias_percent'] for r in records],bins=50,color='#388cb5')
            axes[0].set_xlabel('Volume bias (%)')
            axes[1].hist([r['dice'] for r in records if r['dice'] is not None],bins=50,color='#388cb5')
            axes[1].set_xlabel('Dice')
            counts=Counter('0' if r['intersections']==0 else '1+' for r in records)
            axes[2].bar(['0','1+'],[counts['0'],counts['1+']],color='#b74a44')
            axes[2].set_xlabel('Self-intersection pairs')
            for ax in axes:
                ax.set_ylabel('Observations')
        fig.suptitle('Cohort mesh candidates: every completed observation retained')
        fig.savefig(output/'mesh_metrics.png',dpi=160);fig.savefig(output/'mesh_metrics.pdf');plt.close(fig)
    summary=dict(run_root=str(root),status=progress['status'],planned_subjects=plan['scheduled_subjects'],
        planned_scans=plan['scheduled_scans'],verified_scans=len(records),unresolved_scans=len(remaining),
        flagged_scans=sum(bool(r['qc_flags']) for r in records),
        flag_counts=dict(Counter(flag for r in records for flag in r['qc_flags'].split(';') if flag)),
        topology_and_previous_source_exact=True,artifacts_and_metadata_match_frozen_plan=True,
        unresolved=remaining,quality_flags_do_not_exclude_observations=True,training_adoption=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    if records:
        summary.update(volume_bias_range=[min(r['volume_bias_percent'] for r in records),max(r['volume_bias_percent'] for r in records)],
                       dice_range=[min(r['dice'] for r in records if r['dice'] is not None),max(r['dice'] for r in records if r['dice'] is not None)],
                       total_fitting_seconds=sum(r['fitting_seconds'] for r in records))
        (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    report=['# 一括mesh生成の品質確認','',f"実行状態: {progress['status']}。",'',
        f"予定{plan['scheduled_subjects']}患者・{plan['scheduled_scans']}撮像。保存済み{len(records)}撮像のartifact hash、metadata、1490頂点・2976面、初回template初期化、直前meshの完全な引継ぎを確認した。未完了は{len(remaining)}撮像。",'',
        '同日repeatは日付・UID数値順で全保持する。各撮像のsourceは直前に処理した観測mesh。同日内順序は未記載の補完で、撮像時刻・年齢は変更しない。', '',
        '品質flagは確認候補を示し、cohortからの除外規則には使わない。Dice<0.90、体積差の絶対値>5%、境界平均距離>1仮定mm、表面交差、閉じていないmesh等を記録する。BrainODE/CNNへの採用は未実施。', '',
        '[全数値](mesh_metrics.csv)、[確認結果](summary.json)、[比較図](mesh_metrics.png)。','',
        '| subject / UID | mask体積 | mesh体積 | 体積差 | Dice | 表面交差 | flag |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in records if len(records)<=40 else []:
        dice=f"{r['dice']:.3f}" if r['dice'] is not None else '未算出'
        report.append(f"| {r['subject']} / {r['uid']} | {r['mask_volume']:.1f} | {r['mesh_volume']:.1f} | {r['volume_bias_percent']:+.2f}% | {dice} | {r['intersections']} | {r['qc_flags'] or '—'} |")
    if len(records)>40:
        report += ['', '全撮像の数値とflagはCSVに記録した。分布図は全保存済み観測を含む。', '']
    report+=['','体積は1数値単位=1mmの仮定によるmm³。対象はnative parcellationのlabel75で、TRSAAへの登録は行っていない。', '']
    (output/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps({k:v for k,v in summary.items() if k!='unresolved'},indent=2))


if __name__=='__main__':
    main()
