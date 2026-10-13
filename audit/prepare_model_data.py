"""Verify the completed ADNI1 mesh series and fit training-only PCA150."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lab_shapes import fit_train_pca, patient_indices
from mesh_cohort import ordered_scans, subject_folder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(values.max()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mesh-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root, output, report = args.mesh_root.resolve(), args.output.resolve(), args.report.resolve()
    if not output.is_relative_to(Path('/data5/take/brainode')) or not report.is_relative_to(PROJECT):
        parser.error('arrays belong under data5 and light reports under the home project')
    started = time.monotonic()
    progress = json.loads((root/'progress.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    review_path = PROJECT/'artifacts/mesh_runs/adni1_stage_v3/review/summary.json'
    flag_path = PROJECT/'artifacts/mesh_runs/adni1_stage_v3/flag_review/summary.json'
    review, flag_review = json.loads(review_path.read_text()), json.loads(flag_path.read_text())
    if (progress['status'] != 'completed_candidates' or review['unresolved_scans'] or
            review['run_root'] != str(root) or review['verified_scans'] != plan['scheduled_scans'] or
            not review['topology_and_previous_source_exact'] or
            flag_review['flagged_scans'] != review['flagged_scans']):
        raise ValueError('complete matching mesh and geometry reviews required')
    table = Path(plan['table'])
    if digest(table) != plan['table_sha256']:
        raise ValueError('frozen observation table changed')
    with table.open() as file:
        rows = list(csv.DictReader(file))
    baseline = PROJECT/'artifacts/adni1_stage_baseline'
    with (baseline/'eligible.csv').open() as file:
        original = list(csv.DictReader(file))
    if rows != original:
        raise ValueError('stage observations differ from the frozen mesh table')
    base_summary = json.loads((baseline/'summary.json').read_text())
    template = trimesh.load_mesh(PROJECT/'dataset/Shapes/hippocampus_template_mesh.obj', process=False)
    records, vertices = [], []
    for pid in sorted({r['pid'] for r in rows}):
        series = ordered_scans(r for r in rows if r['pid'] == pid)
        directory = root/subject_folder(pid)
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest['subject'] != pid or len(manifest['scans']) != len(series):
            raise ValueError('mesh series does not cover the patient observations')
        for row, scan in zip(series, manifest['scans']):
            for field, key in [('uid','uid'),('day','day'),('class','diagnosis'),('split','split'),('scan_key','scan_key')]:
                if row[field] != scan[key]:
                    raise ValueError('mesh metadata mismatch')
            if float(row['age']) != scan['age'] or row['parcellation_path'] != scan['source_path']:
                raise ValueError('source metadata mismatch')
            expected_time = (float(row['age'])-base_summary['train_time_scale']['min_age'])/base_summary['train_time_scale']['span_years']
            if abs(float(row['time'])-expected_time) > 1e-12 or float(row['observed_c']) != float(row['class']=='AD'):
                raise ValueError('time or observed c mismatch')
            path = directory/f"{row['uid']}.npz"
            if digest(path) != scan['artifact_sha256'][path.name]:
                raise ValueError('candidate NPZ hash changed')
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['faces'], template.faces)
                mesh_vertices = data['vertices'].copy()
                if mesh_vertices.shape != (1490,3) or not np.isfinite(mesh_vertices).all():
                    raise ValueError('invalid corresponding mesh coordinates')
            vertices.append(mesh_vertices)
            records.append(dict(row, qc_flags=';'.join(scan['qc_flags']), mesh_npz=str(path),
                mesh_npz_sha256=scan['artifact_sha256'][path.name],
                manifest_sha256=digest(directory/'manifest.json'),
                dice=scan['overlap']['dice'], volume_bias_percent=scan['volume_bias_percent']))
    if len(records) != len(rows) or {r['scan_key'] for r in records} != {r['scan_key'] for r in rows}:
        raise ValueError('observations were lost or duplicated')
    vertices = np.stack(vertices)
    pids, splits = np.array([r['pid'] for r in records]), np.array([r['split'] for r in records])
    groups = patient_indices(pids, splits)
    times = np.array([float(r['time']) for r in records], dtype=np.float64)
    print('verified every observation; fitting full-SVD PCA on training scans', flush=True)
    pca, coefficients, fitted = fit_train_pca(vertices, pids, splits, 150)
    reconstructed = pca.inverse_transform(coefficients).reshape(vertices.shape)
    distances = np.linalg.norm(reconstructed-vertices, axis=2)
    errors = distances.mean(axis=1)
    reconstructed_crossings = []
    for reconstructed_vertices in reconstructed:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(reconstructed_vertices),
                                        o3d.utility.Vector3iVector(template.faces))
        reconstructed_crossings.append(len(mesh.get_self_intersecting_triangles()))
    for index, record in enumerate(records):
        record.update(pca_mean_vertex_error_mm=float(errors[index]),
                      pca_vertex_error_p95_mm=float(np.quantile(distances[index],.95)),
                      pca_self_intersection_pairs=reconstructed_crossings[index])
    temporal = {}
    for split in ('train','test'):
        selected = [indices for indices in groups.values() if splits[indices[0]] == split]
        pairs = sum(sum(times[a]!=times[b] for a in indices for b in indices) for indices in selected)
        temporal[split] = dict(subjects=len(selected), scans=int(sum(len(v) for v in selected)),
            subjects_with_temporal_pairs=sum(len(np.unique(times[i]))>1 for i in selected),
            subjects_without_temporal_pairs=sum(len(np.unique(times[i]))==1 for i in selected),
            ordered_distinct_time_pairs=int(pairs),
            one_shot_future_targets=int(sum(np.count_nonzero(times[i]>times[i[0]]) for i in selected)))
    output.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'pca.npz', mean=pca.mean_, components=pca.components_,
        explained_variance=fitted.explained_variance_, explained_variance_ratio=fitted.explained_variance_ratio_,
        singular_values=fitted.singular_values_)
    np.savez_compressed(output/'observations.npz', vertices=vertices, coefficients=coefficients,
        faces=template.faces, times=times, conditions=np.array([float(r['observed_c']) for r in records]),
        pids=pids, splits=splits, uids=np.array([r['uid'] for r in records]),
        days=np.array([r['day'] for r in records]), ages=np.array([float(r['age']) for r in records]),
        qc_flags=np.array([r['qc_flags'] for r in records]))
    with (report/'observations.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = dict(cohort='ADNI1', observations=len(records), dimensions=150, vertex_count=1490,
        train_only_pca=True, pca_fit_scans=int(sum(splits=='train')), pca_fit_policy='all train observations; no whitening; full SVD',
        explained_variance_ratio_sum=float(fitted.explained_variance_ratio_.sum()),
        reconstruction_mean_vertex_error_mm={split:distribution(errors[splits==split]) for split in ('train','test')},
        pca_reconstructed_scans_with_self_intersection=int(sum(v>0 for v in reconstructed_crossings)),
        pca_self_intersection_pair_max=max(reconstructed_crossings),
        temporal_counts=temporal, time_scale=base_summary['train_time_scale'],
        flagged_observations_retained=int(sum(bool(r['qc_flags']) for r in records)), observations_excluded=0,
        quality_policy='initial observed baseline retains all reviewed finite shared-topology candidates; residual quality flags remain and are reported separately',
        spatial_unit_assumption='one numerical world-coordinate unit interpreted as one mm; source headers remain unknown',
        mesh_root=str(root), data_root=str(output),
        source_hashes={str(p):digest(p) for p in (table,baseline/'summary.json',review_path,flag_path)},
        data_sha256={p.name:digest(p) for p in output.glob('*.npz')},
        elapsed_seconds=time.monotonic()-started, future_prediction_evaluated=False,
        public_code_difference='PCA fitted on all training observations rather than first observations only')
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for split in ('train','test'):
        ax.hist(errors[splits==split],bins=40,alpha=.55,label=split)
    ax.set_xlabel('PCA150 reconstruction mean vertex distance (assumed mm)')
    ax.set_ylabel('Observations');ax.legend()
    fig.savefig(report/'reconstruction_error.png',dpi=160);fig.savefig(report/'reconstruction_error.pdf');plt.close(fig)
    text=['# ADNI1 PCA150と学習用観測データ','',
        '全468患者・1,405撮像を対応付け、元meshと同じ頂点順序・face、metadata、artifact hashを確認した。73撮像の品質flagも保持する。', '',
        f"trainの全1,150撮像だけでPCA150をfitし、保持した分散は{summary['explained_variance_ratio_sum']*100:.4f}%。testの255撮像には同じ基底を適用した。",'',
        f"再構成後の自己交差ありは{summary['pca_reconstructed_scans_with_self_intersection']}撮像。元meshの修復や除外は行っていない。",'',
        f"再構成誤差: `{json.dumps(summary['reconstruction_mean_vertex_error_mm'])}`。距離単位は元headerの数値をmmとする仮定。",'',
        f"時間対: `{json.dumps(temporal)}`。時間対を持たない患者もPCAデータに含まれる。",'',
        '公開コードとの差はPCAのfit対象を初回撮像だけから全train観測へ広げる点。同日repeatは別観測のまま保持し、時刻をずらさない。', '',
        '初期観測baselineは品質確認済みの全候補を使用する。局所自己交差と低Diceの残存を記録し、予測評価をflag別にも報告する。これは連続cを推定するCNN/Algorithm 2の学習完了ではない。','',
        f"配列: `{output}`。数値NPZで保存し、pickleを読み込まない。",'',
        '[設定・数値](summary.json)、[撮像台帳と再構成誤差](observations.csv)、[誤差分布](reconstruction_error.png)。','']
    (report/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
