"""Export frozen ADNI1 results, segmentations and forecasts for portable 3D review.

This is a display-only export. It performs no training, mesh repair, image
resampling, or alteration of original research data and experiment snapshots.
"""

import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / 'artifacts/training_runs/adni1_continuous_v1'
INITIAL = PROJECT / 'artifacts/training_runs/adni1_observed_v1'
OUTPUT = PROJECT / 'artifacts/review_bundles/adni1_RID352'
PAPER = ('https://proceedings.neurips.cc/paper_files/paper/2025/file/'
         'abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf')
SLICER_DOC = ('https://slicer.readthedocs.io/en/latest/user_guide/'
              'data_loading_and_saving.html#specifying-the-coordinate-system-in-model-files')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def expect_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError('Frozen input changed: ' + str(path))
    return actual


def write_mesh(prefix, vertices, faces):
    """Double precision RAS coordinates, original vertex/face ordering."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid mesh')
    vtk, obj = prefix.with_suffix('.vtk'), prefix.with_suffix('.obj')
    with vtk.open('w') as file:
        file.write('# vtk DataFile Version 3.0\nBrainODE review. SPACE=RAS\nASCII\nDATASET POLYDATA\n')
        file.write(f'POINTS {len(vertices)} double\n')
        for vertex in vertices:
            file.write(' '.join(format(float(x), '.17g') for x in vertex) + '\n')
        file.write(f'POLYGONS {len(faces)} {4 * len(faces)}\n')
        for face in faces:
            file.write('3 ' + ' '.join(str(int(x)) for x in face) + '\n')
    with obj.open('w') as file:
        file.write('# BrainODE review. SPACE=RAS\n')
        for vertex in vertices:
            file.write('v ' + ' '.join(format(float(x), '.17g') for x in vertex) + '\n')
        for face in faces:
            file.write('f ' + ' '.join(str(int(x) + 1) for x in face) + '\n')
    restored = trimesh.load_mesh(obj, process=False)
    np.testing.assert_array_equal(restored.faces, faces)
    np.testing.assert_allclose(restored.vertices, vertices, atol=1e-12, rtol=0)
    # Read the primary VTK file back independently of the OBJ importer.
    lines = vtk.read_text().splitlines()
    assert 'SPACE=RAS' in lines[1] and lines[3] == 'DATASET POLYDATA'
    assert lines[4] == f'POINTS {len(vertices)} double'
    loaded_vertices = np.array([[float(x) for x in line.split()]
                               for line in lines[5:5 + len(vertices)]])
    assert lines[5 + len(vertices)] == f'POLYGONS {len(faces)} {4 * len(faces)}'
    loaded_faces = np.array([[int(x) for x in line.split()[1:]]
                            for line in lines[6 + len(vertices):]])
    np.testing.assert_array_equal(loaded_vertices, vertices)
    np.testing.assert_array_equal(loaded_faces, faces)
    return vtk, obj


def voxel_boundary(mask, affine):
    """Exact external faces of foreground voxel cells; no smoothing."""
    # Corner offsets are integers in a grid twice as fine as voxel indices.
    directions = [
        ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
        ((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
        ((0, 1, 0), [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
        ((0, 0, -1), [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]),
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
    ]
    corners, corner_indices, faces = [], {}, []
    shape = np.array(mask.shape)
    for center in np.argwhere(mask):
        for direction, offsets in directions:
            neighbor = center + direction
            if np.all(neighbor >= 0) and np.all(neighbor < shape) and mask[tuple(neighbor)]:
                continue
            quad = []
            for offset in offsets:
                corner = tuple(2 * center + offset)
                if corner not in corner_indices:
                    corner_indices[corner] = len(corners)
                    corners.append(corner)
                quad.append(corner_indices[corner])
            faces.extend([(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])])
    vertices = nib.affines.apply_affine(affine, np.array(corners, dtype=np.float64) / 2)
    faces = np.array(faces, dtype=np.int64)
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    volume = trimesh.Trimesh(vertices, faces, process=False).volume
    expected = int(mask.sum()) * abs(np.linalg.det(affine[:3, :3]))
    np.testing.assert_allclose(volume, expected, rtol=1e-10, atol=1e-8)
    return vertices, faces


def scan_data(row):
    path = Path(row['mesh_npz'])
    expect_hash(path, row['mesh_npz_sha256'])
    original_manifest = json.loads((path.parent / 'manifest.json').read_text())
    record = next(s for s in original_manifest['scans'] if s['uid'] == row['uid'])
    expect_hash(Path(row['parcellation_path']), record['source_sha256'])
    with np.load(path, allow_pickle=False) as data:
        saved = {key: data[key].copy() for key in data.files}
    rotation = saved['icp_matrix'][:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0)
    native_to_canonical = saved['icp_matrix'].astype(np.float64).copy()
    native_to_canonical[:3, 3] -= rotation @ saved['centroid_header_world']
    canonical_to_native = np.linalg.inv(native_to_canonical)
    original = nib.load(row['parcellation_path'])
    np.testing.assert_array_equal(original.affine, saved['affine'])
    mask = (np.asarray(original.dataobj) == 75).astype(np.uint8)
    return dict(row=row, saved=saved, image=original, mask=mask,
                n2c=native_to_canonical, c2n=canonical_to_native,
                source_sha256=record['source_sha256'])


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_output = OUTPUT / 'data'
    provenance = OUTPUT / 'provenance'
    data_output.mkdir(exist_ok=True)
    provenance.mkdir(exist_ok=True)
    config = json.loads((REPORT / 'deterministic_pool_config.json').read_text())
    results = json.loads((REPORT / 'RESULTS.json').read_text())
    csv_rows = read_csv(PROJECT / 'artifacts/model_data/adni1_pca150_v1/observations.csv')
    uid_rows = {row['uid']: row for row in csv_rows}
    source, target = scan_data(uid_rows['34537']), scan_data(uid_rows['176917'])
    assert source['row']['pid'] == target['row']['pid'] == 'ADNI:RID:352'
    assert source['row']['split'] == target['row']['split'] == 'test'
    observations_path = Path(config['data_root']) / 'observations.npz'
    expect_hash(observations_path, config['data_sha256']['observations.npz'])
    with np.load(observations_path, allow_pickle=False) as observations:
        for scan in (source, target):
            matches = np.flatnonzero(observations['uids'] == scan['row']['uid'])
            assert len(matches) == 1
            np.testing.assert_array_equal(observations['vertices'][matches[0]], scan['saved']['vertices'])
    initial_path = Path('/data5/take/brainode/experiments/adni1_observed_v1/future_predictions.npz')
    current_path = Path(config['output']) / 'evaluate/future_predictions.npz'
    prediction_hashes = {
        'initial': expect_hash(initial_path, '8a41155e961a8bae8a07a0f6753649fce63d2bd5232e223a9e33dfd5c19bb48b'),
        'algorithm2': expect_hash(current_path, results['geometry']['prediction_file_sha256']),
    }
    predictions = {}
    for name, path in [('initial', initial_path), ('algorithm2', current_path)]:
        with np.load(path, allow_pickle=False) as saved:
            idx = np.flatnonzero((saved['source_uids'] == '34537') & (saved['target_uids'] == '176917'))
            assert len(idx) == 1 and saved['pids'][idx[0]] == source['row']['pid']
            np.testing.assert_array_equal(saved['faces'], source['saved']['faces'])
            predictions[name] = saved['vertices'][idx[0]].astype(np.float64)
    initial_metric = next(r for r in read_csv(INITIAL / 'future_metrics.csv')
                          if r['source_uid'] == '34537' and r['target_uid'] == '176917')
    current_metric = next(r for r in read_csv(REPORT / 'evaluate/future_metrics.csv')
                          if r['source_uid'] == '34537' and r['target_uid'] == '176917')
    source_vertices = source['saved']['vertices'].astype(np.float64)
    target_vertices = target['saved']['vertices'].astype(np.float64)
    errors = {}
    for key, vertices in [('static', source_vertices), *predictions.items()]:
        canonical_error = float(np.linalg.norm(vertices - target_vertices, axis=1).mean())
        native_error = float(np.linalg.norm(nib.affines.apply_affine(source['c2n'], vertices)
                                          - nib.affines.apply_affine(source['c2n'], target_vertices), axis=1).mean())
        np.testing.assert_allclose(native_error, canonical_error, atol=1e-12, rtol=0)
        expected = (initial_metric['prediction_mean_vertex_euclidean'] if key == 'initial'
                    else current_metric['prediction_mean_vertex_euclidean'] if key == 'algorithm2'
                    else current_metric['static_source_mean_vertex_euclidean'])
        np.testing.assert_allclose(canonical_error, float(expected), atol=1e-10, rtol=0)
        errors[key] = canonical_error
    segmentations, models, model_arrays, scan_records = [], [], {}, []
    for key, scan, prefix, color in [
        ('baseline', source, '01_baseline', [0.63, 0.68, 0.74]),
        ('future', target, '04_future_reference', [0.20, 0.70, 0.48]),
    ]:
        # Both scans are placed in baseline-native RAS, using saved rigid ICP only.
        native_to_display = source['c2n'] @ scan['n2c']
        affine = (scan['image'].affine.copy() if key == 'baseline'
                  else native_to_display @ scan['image'].affine)
        header = scan['image'].header.copy()
        header.set_data_dtype(np.uint8)
        header.set_slope_inter(1, 0)
        header.set_intent('none')
        header.set_xyzt_units('mm', 'unknown')
        header['descrip'] = b'Hippo_L75->1; display mm assumed; no voxel resampling'
        image = nib.Nifti1Image(scan['mask'], affine, header=header)
        image.set_qform(affine, code=0)
        image.set_sform(affine, code=2)
        mask_path = data_output / f'{prefix}_segmentation.nii.gz'
        nib.save(image, mask_path)
        restored = nib.load(mask_path)
        np.testing.assert_array_equal(np.asarray(restored.dataobj), scan['mask'])
        if key == 'baseline':
            np.testing.assert_array_equal(restored.affine, scan['image'].affine)
        indices = np.argwhere(scan['mask'])
        affine_error = float(np.linalg.norm(nib.affines.apply_affine(restored.affine, indices)
                                           - nib.affines.apply_affine(affine, indices), axis=1).max())
        assert affine_error < 3e-5
        segmentations.append(dict(key=key + '_mask', name=prefix + '_segmentation',
                                  file=str(mask_path.relative_to(OUTPUT)), color=color,
                                  default_visible=(key == 'baseline')))
        v, f = voxel_boundary(scan['mask'], affine)
        vtk, obj = write_mesh(data_output / f'{prefix}_segmentation_voxel_boundary', v, f)
        boundary_key = key + '_voxel_boundary'
        models.append(dict(key=boundary_key, name=prefix + '_segmentation_voxel_boundary',
                           vtk=str(vtk.relative_to(OUTPUT)), obj=str(obj.relative_to(OUTPUT)),
                           color=color, opacity=0.30, default_visible=(key == 'baseline'),
                           role='Exact voxel-cell boundary for display only; not a fitted training mesh',
                           vertices=len(v), faces=len(f)))
        model_arrays[boundary_key] = (v, f)
        scan_records.append(dict(uid=scan['row']['uid'], day=scan['row']['day'],
            diagnosis=scan['row']['class'], age=float(scan['row']['age']), split=scan['row']['split'],
            mask_voxels=int(scan['mask'].sum()), full_grid_shape=list(scan['mask'].shape),
            foreground_bounding_box_min_ijk=indices.min(axis=0).tolist(),
            foreground_bounding_box_max_ijk=indices.max(axis=0).tolist(),
            foreground_bounding_box_shape=(indices.max(axis=0)-indices.min(axis=0)+1).tolist(),
            fitted_mesh_dice=float(scan['row']['dice']),
            fitted_mesh_volume_bias_percent=float(scan['row']['volume_bias_percent']),
            pca_reconstruction_mean_vertex_error=float(scan['row']['pca_mean_vertex_error_mm']),
            source_spatial_units=scan['image'].header.get_xyzt_units()[0],
            original_parcellation=scan['row']['parcellation_path'],
            original_parcellation_sha256=scan['source_sha256'],
            original_mesh_npz=scan['row']['mesh_npz'], original_mesh_sha256=scan['row']['mesh_npz_sha256'],
            original_affine=scan['image'].affine.tolist(), exported_affine=restored.affine.tolist(),
            native_to_canonical=scan['n2c'].tolist(), native_to_display=native_to_display.tolist(),
            foreground_world_coordinate_roundtrip_max_error=affine_error,
            mask_values_exact=True, voxel_resampling=False))
    for key, prefix, canonical, color, opacity, visible in [
        ('baseline_mesh', '02_baseline_fitted_mesh', source_vertices, [0.12, 0.40, 0.86], 0.75, True),
        ('prediction', '03_future_prediction_algorithm2', predictions['algorithm2'], [0.97, 0.44, 0.10], 0.65, False),
        ('future_mesh', '05_future_observed_mesh', target_vertices, [0.10, 0.66, 0.32], 0.60, False),
        ('initial_prediction', '06_future_prediction_initial_ode', predictions['initial'], [0.60, 0.27, 0.78], 0.55, False),
    ]:
        vertices = nib.affines.apply_affine(source['c2n'], canonical)
        np.testing.assert_allclose(nib.affines.apply_affine(source['n2c'], vertices), canonical,
                                   atol=1e-11, rtol=0)
        faces = source['saved']['faces']
        vtk, obj = write_mesh(data_output / prefix, vertices, faces)
        np.testing.assert_allclose(trimesh.Trimesh(vertices, faces, process=False).volume,
                                   trimesh.Trimesh(canonical, faces, process=False).volume,
                                   atol=1e-8, rtol=1e-10)
        models.append(dict(key=key, name=prefix, vtk=str(vtk.relative_to(OUTPUT)),
                           obj=str(obj.relative_to(OUTPUT)), color=color, opacity=opacity,
                           default_visible=visible, vertices=len(vertices), faces=len(faces),
                           role=('Fitted observed mesh before PCA' if key.endswith('_mesh')
                                 else 'Retrospective future forecast from baseline; no target supplied')))
        model_arrays[key] = (vertices, faces)
    modes = dict(segmentation=['baseline_voxel_boundary', 'baseline_mesh'],
                 change=['baseline_mesh', 'prediction'],
                 prediction=['future_voxel_boundary', 'future_mesh', 'prediction'],
                 models=['initial_prediction', 'prediction', 'future_mesh'])
    manifest = dict(schema_version=1, experiment='adni1_continuous_v1',
        pid=source['row']['pid'], source_uid='34537', target_uid='176917',
        source_day=source['row']['day'], target_day=target['row']['day'],
        horizon_years=float(current_metric['horizon_years']),
        selection='Reuses the previously fixed preview patient, selected by median initial-baseline error; not selected by improvement of the new model',
        coordinate_system='Baseline native-parcellation RAS for all exported files',
        display_transform='Canonical meshes -> baseline inverse ICP + boundary centroid; future segmentation -> its native-to-canonical -> baseline canonical-to-native',
        canonical_to_display=source['c2n'].tolist(),
        units='Numeric 1 interpreted as 1 mm. Exported NIfTI units set to mm as a display assumption; original units unknown.',
        limitations=['Left hippocampus label75 only', 'Rigid shape alignment only; no verified TRSAA or whole-brain registration',
                     'Forecast is a held-out retrospective prediction to 2010-04-28, not an extrapolation to an unobserved date',
                     'Self-intersections remain; no mesh repair'],
        segmentations=segmentations, models=models, display_modes=modes, scans=scan_records,
        example_errors=errors, prediction_sha256=prediction_hashes,
        example_prediction_self_intersection_pairs=int(current_metric['prediction_self_intersection_pairs']),
        example_prediction_affected_area_percent=float(current_metric['prediction_affected_area_percent']))
    write_json(OUTPUT / 'bundle_manifest.json', manifest)
    shutil.copyfile(PROJECT / 'audit/review_bundle_slicer.py', OUTPUT / 'open_in_slicer.py')
    for origin, name in [(REPORT / 'config.json', 'sampling_and_ode_config.json'),
                         (REPORT / 'deterministic_pool_config.json', 'cnn_and_evaluation_config.json'),
                         (REPORT / 'RESULTS.json', 'full_results.json'),
                         (REPORT / 'evaluate/future_metrics.csv', 'all_future_metrics.csv'),
                         (REPORT / 'evaluate/latest_visit_metrics.csv', 'latest_future_metrics.csv'),
                         (REPORT / 'cnn/cognition_predictions.csv', 'cognition_predictions.csv')]:
        shutil.copyfile(origin, provenance / name)
    write_json(provenance / 'example_metrics.json', dict(initial=initial_metric, algorithm2=current_metric))
    shutil.copyfile(Path(__file__), provenance / 'export_review_bundle.py')
    shutil.copyfile(PROJECT / 'audit/review_bundle_slicer.py', provenance / 'review_bundle_slicer.py')
    # A self-contained browser preview supplies exact segmentation voxel boundaries.
    fig = go.Figure()
    for record in models:
        v, f = model_arrays[record['key']]
        color = 'rgb(' + ','.join(str(round(c * 255)) for c in record['color']) + ')'
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2], name=record['name'], color=color,
            opacity=record['opacity'], visible=record['default_visible'], showlegend=True,
            hovertemplate=record['name'] + '<br>RAS=(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'))
    labels = {'segmentation': '① segmentationと観測mesh', 'change': '② 初回meshと約4年後の予測',
              'prediction': '③ 予測と実際の将来形状', 'models': '④ 初期モデルと追加モデルの比較'}
    buttons = [dict(label=labels[key], method='update',
                    args=[{'visible': [r['key'] in selected for r in models]},
                          {'title': labels[key] + ' — RID352 左海馬'}]) for key, selected in modes.items()]
    bounds = np.concatenate([v for v, f in model_arrays.values()])
    lower, upper = bounds.min(0) - 2, bounds.max(0) + 2
    fig.update_layout(title=labels['segmentation'] + ' — RID352 左海馬',
        scene=dict(aspectmode='data',
                   xaxis=dict(title='R (仮定mm)', range=[lower[0], upper[0]]),
                   yaxis=dict(title='A (仮定mm)', range=[lower[1], upper[1]]),
                   zaxis=dict(title='S (仮定mm)', range=[lower[2], upper[2]])),
        updatemenus=[dict(buttons=buttons, direction='down', x=0, y=1.11)],
        margin=dict(t=100, l=10, r=10, b=10), height=740, legend=dict(x=1.01, y=1))
    viewer = fig.to_html(full_html=False, include_plotlyjs=True)
    intro = ('<h1>左海馬の3D比較</h1><p>2006-04-07 → 2010-04-28（約4.06年）。'
             'ドラッグで回転、ホイールで拡大。上のメニューで比較を選択。'
             '青＝初回観測mesh、橙＝連続cモデルの将来予測、緑＝将来の実測mesh、'
             '灰＝segmentationのvoxel境界。全形状を初回parcellationのRAS位置で表示。</p>'
             '<p>この1例の予測誤差：初期0.4458、追加0.4467 仮定mm。'
             '全体の結果は<a href="REPORT.html">設定・スコア比較</a>を参照。</p>')
    (OUTPUT / 'viewer.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8">'
        '<title>BrainODE 3D review</title><body style="font-family:sans-serif">' + intro + viewer +
        '</body></html>', encoding='utf-8')
    latest = results['latest']['patient_macro_mean']
    all_values = results['all']['patient_macro_mean']
    score_rows = [
        ['初回meshをそのまま将来に使用', latest['static_source_mean_vertex_euclidean'], all_values['static_source_mean_vertex_euclidean'], '同じtest・同じtarget'],
        ['初期BrainODE（観測c・100 epochs）', latest['initial_observed_ode_euclidean'], all_values['initial_observed_ode_euclidean'], '同じtest・同じtarget'],
        ['連続c BrainODE（120 epochs・Algorithm 2）', latest['prediction_mean_vertex_euclidean'], all_values['prediction_mean_vertex_euclidean'], '同じtest・同じtarget'],
        ['論文 Table 2・不規則1-shot海馬', 0.492, '', '参考値：異なるcohort・segmentation・time・設定'],
    ]
    with (OUTPUT / 'score_comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['method', 'latest_patient_macro_euclidean', 'all_future_patient_macro_euclidean', 'comparison_scope'])
        writer.writerows(score_rows)
    table_rows = [[r[0], f'{r[1]:.4f}', f'{r[2]:.4f}' if r[2] != '' else '—', r[3]] for r in score_rows]
    settings = [
        ('データ', 'ADNI1のみ、468患者・1,405撮像。CN 712、AD 693。MCI撮像を除外、MCI歴のある患者全体は除外しない。'),
        ('年齢とtime', '有限年齢のCN/ADを使用。65–95歳の固定制限なし。t=(age−55.7288)/37.2931。最小値と幅はtrainだけから算出、testでは再fit・clipしない。'),
        ('患者分割', 'train 389患者/1,150撮像、test 79患者/255撮像。seed42の既存患者分割を固定。同日重複・単時点患者も保持。'),
        ('segmentation', '既存parcellationの左海馬Hippo_L=label75を使用。PointNetでsegmentationを新規推定したものではない。'),
        ('mesh作成', '公開PointNetOptを各撮像で初期化して5,000 updates。初回はmask体積でtemplateを初期化、後続は前撮像meshを引き継ぐ。1,490頂点・2,976三角形。剛体ICP、追加拡縮なし。'),
        ('PCA', 'train全1,150撮像で150成分をfit、whiteningなし、説明分散97.38%。PCAをtestへ再fitしない。'),
        ('初期BrainODE', '公開ODEFuncWithAttention(150,1)を変更せず使用。RK4、100 epochs、AdamW 5e−4、decay0.01、2患者/batch、StepLR20/0.5。'),
        ('ODE loss・augmentation', '全ての異なる時間の順序付きforward/backward対のPCA二乗誤差を患者内平均、患者等重み。各患者・epoch共通の実頂点scale[0.98,1.02]。'),
        ('連続cの教師', 'trainのCN→AD 2患者/4端点対。式(4)：CN側forwardとAD側backward予測を1/2ずつ平均。各対9内部時点、計36 pseudo。cは時間割合。MMSE/CDR/MCI=0.5は使わない。'),
        ('CNN入力', 'PCA meshから固定128³のcenter occupancy。pitch0.5、origin(−31,−34,−33)。個別の再中心化・拡縮なし。'),
        ('CNN', 'Conv3D 1→16→32→64、kernel3、ReLU・MaxPool2、global average pooling、sigmoid。soft BCE、AdamW5e−4、batch16、StepLR10/0.5、30 epochs。実測1,150+pseudo36を毎epoch1回ずつ。'),
        ('追加ODE', '初期100 epoch checkpointからfresh optimizerで20 epochs追加、合計120。全299 temporal train患者を維持。converterのlossは実測1/2＋pseudo trajectory平均1/2。'),
        ('将来予測', 'Algorithm 2、N=10等時間間隔、毎stepで現在の予測形状からcを再推定。初回撮像のみを入力し、将来診断・将来meshを入力しない。'),
        ('checkpoint・計算', '固定最終epoch、testで設定選択なし。seed42、FP32、TF32 off、deterministic。物理GPU4/5の2枚。'),
        ('未公開部分の補完', 'CNNの層/loss/voxel、pseudo点数・混合、追加20 epochs、N=10は論文未記載部分の実装設定。公開コードの初回だけPCAとは異なり、全train撮像でPCAをfit。'),
    ]
    score_table = markdown_table(['方法', '最終時点：誤差↓', '全将来時点：誤差↓', '比較範囲'], table_rows)
    settings_md = '\n\n'.join('**' + name + '**：' + text for name, text in settings)
    doc = f'''# 今回の設定と3D確認用データ

ADNI1の左海馬を対象に、segmentationから観測meshを作り、初回撮像から将来meshを予測した。初期BrainODEは静的meshより誤差が小さいが、連続c・pseudo追加後の形状誤差は改善していない。

## スコア

{score_table}

単位は仮定mm。対応する頂点間のEuclidean距離をmesh内で平均し、患者内でtargetを平均した後、患者を等重みで平均する。元の観測meshを正解とする。最終時点は60患者・62撮像、全将来時点は60患者・175撮像。同日重複の最終撮像も含む。

連続cモデルは静的mesh比で29.37%誤差が小さい。一方、初期モデル比で1.74%大きい（差+0.00740、患者bootstrap95%区間+0.00231〜+0.01245）。改善は19/60患者。同じN=10・observed cでも追加ODEは0.43281となり、劣化の大半は追加最適化とpseudoの組合せで生じる。pseudo単独の因果効果を分離した実験ではない。

論文Table 2の0.492 mmより今回の0.4332は数値上小さいが、論文はADNI/AIBL/OASIS2/OASIS3を含み、今回はADNI1のみ。患者、segmentation、時間係数、単位の確認状態、PCA、学習配分等が異なるため、論文を上回ったとは結論しない。公開コードの座標MAE（今回0.1643）はEuclidean距離と別指標であり、0.492とは比較しない。

CN/AD識別はtest79患者・255撮像で撮像正解率87.45%、balanced accuracy87.05%、AUC0.9292。患者内正解率を患者等重みで平均すると83.73%。論文Table 3のc推定器正解率89.1%は参考値で、評価集団・推論段階が一致しない。今回の診断識別は形状予測誤差とは別の評価である。

pseudo教師cのtrain MAEは0.22948、一律0.5の0.22222を下回らない。train converterは2患者、test converterは0患者。連続cの臨床的な精度やCN→ADへの将来変化は検証できていない。

## 固定設定

{settings_md}

## 3Dファイルの対象

既存の代表previewを再使用：test患者RID352。初回2006-04-07（UID34537、77.1397歳、CN）から2010-04-28（UID176917、81.2歳、CN）への約4.06年後予測。初期baseline誤差の中央値を基準に選ばれた既存例を維持し、新モデルの改善度による選択は行っていない。

この1例のEuclidean誤差：静的mesh0.81419、初期ODE0.44577、連続cモデル0.44671 仮定mm。この1例の値と全患者平均は別である。

segmentationは原parcellationのlabel75だけをforeground1として抽出した256³ NIfTI。maskのvoxel値は完全保持し、resampling・成分削除・平滑化なし。初回ROIのbounding boxは26×40×18 voxels。128³ CNN入力は別の固定voxel gridであり、segmentation原画像と同じサイズである必要はない。

初回観測meshのmask Diceは0.92940、体積差−0.504%。将来観測meshはDice0.96175、体積差+0.443%。完全一致ではないが、この2撮像のfitは大きな体積の不一致を示していない。全体のfitはDice0.891〜0.973、体積差−3.53〜+2.49%。境界meshとPCA/予測meshは区別する。

全ファイルを初回parcellationのnative RASに置いた。meshは保存済みinverse ICPとcentroidで戻し、将来segmentationはその撮像の保存済み剛体ICPを通して初回の表示座標へ置いた。形状の拡縮はない。将来maskのaffineだけを変更し、voxel値は変更していない。これはshape alignmentであり、TRSAAやMRI全体のregistrationを確認したものではない。

元headerの空間単位はunknown。座標の数値1を1mmとする実験の仮定を維持し、出力NIfTIの単位は表示用にmmと明示した。予測は2010-04-28を将来targetとするretrospective評価であり、未観測の日付への追加推論は行っていない。

## 形式とファイル

- `data/01_baseline_segmentation.nii.gz`：初回の左海馬binary segmentation。背景0、左海馬1。
- `data/02_baseline_fitted_mesh.vtk`：初回のPointNet fitによる観測mesh。
- `data/03_future_prediction_algorithm2.vtk`：連続cモデルの約4年後予測。
- `data/04_future_reference_segmentation.nii.gz`：将来の実際のsegmentationを初回の表示座標へ整列。
- `data/05_future_observed_mesh.vtk`：将来の実際の観測mesh、予測の比較対象。
- `data/06_future_prediction_initial_ode.vtk`：初期100 epochモデルの同じ将来予測。
- `*_segmentation_voxel_boundary.vtk`：NIfTIのvoxel境界をそのまま立体表示する補助model。学習用PointNet meshではない。
- `.obj`：各meshの同内容の汎用形式。Slicerでは座標規約を明記した`.vtk`を推奨。
- `viewer.html`：オフラインで回転・拡大できる3D比較。
- `REPORT.html`：設定とスコアの閲覧用。
- `score_comparison.csv`：全体スコア比較。`provenance/`に固定設定、全targetの結果、生成コードを同梱。

## Slicerで開く

ZIPを任意のローカルフォルダに展開する。ファイルを移動する際はフォルダ構成を維持する。

自動表示：SlicerのView → Python Interactorを開き、次を実行する。パスは展開した場所へ置換する。

```python
exec(open(r"C:/path/adni1_RID352/open_in_slicer.py", encoding="utf-8").read())
```

ファイル選択画面で、同じフォルダの`bundle_manifest.json`を選ぶ。初回segmentationのvoxel境界（灰）と観測mesh（青）が表示される。Python consoleから次の切替が可能。

初回は全比較meshの範囲を基準に、左側からの平行投影で海馬を拡大表示する。表示切替ではカメラを変更しない。同じ倍率で比較し、形状そのものの拡縮は行わない。

```python
show_brainode("segmentation")  # 初回segmentationと観測mesh
show_brainode("change")        # 初回meshと将来予測
show_brainode("prediction")    # 将来予測と実際の将来形状
show_brainode("models")        # 初期・追加モデルと将来観測mesh
show_brainode("baseline")      # 初回meshだけ
show_brainode("forecast")      # 将来予測だけ
show_brainode("reference")     # 実際の将来meshだけ
zoom_brainode(1.5)             # 現在の視点のまま表示倍率を1.5倍にする
frame_brainode()               # 共通の左側視点・初期倍率へ戻す
```

Dataの目のアイコンでも各modelの表示を切り替えられる。既存sceneは消さない。必要ならSaveでsceneを`.mrb`へ保存すると、以後は1ファイルで再表示できる。

手動表示：File → Add Dataで`.vtk`をModelとして読み込む。NIfTIはSegmentationとして読み込むと2D maskを確認でき、SegmentationsのCreate closed surface representation / Show 3Dで立体表示できる。灰色の補助`.vtk`を使えばvoxel境界を直接表示できる。

## 残る形状の問題と検証範囲

連続cモデルの予測175件のうち103件に自己交差がある。最大影響面積1.633%。本例も21 triangle-pairs、影響面積0.1217%の交差が残る。今回の表示exportは修復しない。共有頂点番号による距離評価は、解剖学的な頂点対応やsurfaceの妥当性を単独で保証しない。

原データ・固定予測のSHA256、NIfTI voxel値、affineによる座標変換、VTK/OBJ再読込、mesh体積、剛体変換前後の距離を検証した。Slicer本体はこの実行環境にないため、添付loaderのGUI実行は未確認。対応する公式APIと座標規約に従っている。詳細は`verification.json`、`bundle_manifest.json`、`checksums.sha256`。

根拠：[BrainODE論文 Table 2・3、式(4)、Algorithm 1・2、Appendix]({PAPER})、[Slicer対応形式・座標規約]({SLICER_DOC})、[Slicer Python APIの使用例](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#segmentations)。
'''
    (OUTPUT / 'REPORT.md').write_text(doc, encoding='utf-8')
    settings_html = ''.join(f'<tr><th>{html.escape(name)}</th><td>{html.escape(text)}</td></tr>' for name, text in settings)
    rows_html = ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in table_rows)
    # Keep the full Markdown report readable offline without an external renderer.
    report_html = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>BrainODE 設定とスコア</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.7}}table{{border-collapse:collapse;width:100%;margin:24px 0}}th,td{{border:1px solid #ccd1da;padding:10px;text-align:left}}th{{background:#edf2f8}}pre{{white-space:pre-wrap;background:#f6f7f9;padding:20px}}a{{color:#195caf}}</style>
<h1>ADNI1・左海馬：固定設定と結果</h1><p>初期モデルより連続c追加後の形状誤差は1.74%大きく、改善していない。静的mesh比では29.37%小さい。</p>
<p><a href="viewer.html">回転できる3D比較を開く</a> · <a href="REPORT.md">完全なMarkdownレポート</a> · <a href="score_comparison.csv">比較CSV</a></p>
<table><tr><th>方法</th><th>最終時点：誤差↓</th><th>全将来：誤差↓</th><th>比較範囲</th></tr>{rows_html}</table>
<p>単位：仮定mm。最終時点60患者62撮像、全将来60患者175撮像。患者macroの対応頂点Euclidean距離。論文とは評価条件が異なり、数値上小さくても優位性は結論できない。</p>
<p>CN/AD分類：正解率87.45%、AUC0.9292。論文Table 3の89.1%は別条件での参考値。連続c教師MAE0.22948は定数0.5の0.22222を下回らない。予測103/175件に自己交差が残る。</p>
<table><tr><th>設定</th><th>今回の実装</th></tr>{settings_html}</table>
<p>根拠：<a href="{PAPER}">BrainODE論文</a>、<a href="{SLICER_DOC}">Slicer公式形式・座標規約</a>。</p>
<h2>ファイル・表示方法・検証の全記録</h2><pre>{html.escape(doc)}</pre></html>'''
    (OUTPUT / 'REPORT.html').write_text(report_html, encoding='utf-8')
    (OUTPUT / 'README.txt').write_text('BrainODE ADNI1 左海馬 3D確認用パッケージ\n\n'
        '1. ZIPを展開する。\n2. REPORT.htmlで設定・結果を確認する。\n'
        '3. viewer.htmlはブラウザでオフライン3D表示。\n'
        '4. SlicerはREPORT.mdの手順でopen_in_slicer.pyを実行するか、data/の.vtkをModelとして開く。\n'
        'NIfTIは3D segmentation、VTK/OBJは3D surface。全て初回parcellationのRAS座標。\n'
        '対象は左海馬、2006-04-07→2010-04-28。予測と実際の将来形状は別ファイル。\n'
        '元の空間単位はunknown。1=1mmという表示仮定を維持。自己交差は未修復。\n', encoding='utf-8')
    verification = dict(passed=True, original_source_hashes_verified=True,
        frozen_observations_and_prediction_hashes_verified=True,
        original_source_files_unchanged=True, binary_mask_values_exact=True,
        baseline_mask_affine_exact=True, voxel_resampling=False,
        rigid_rotation_verified=True, mesh_coordinate_roundtrip_tolerance=1e-11,
        vtk_and_obj_readback_verified=True, vertex_and_face_order_preserved=True,
        exact_voxel_boundary_volume_verified=True, learned_mesh_volume_preserved=True,
        distance_and_saved_example_metrics_verified=True,
        original_segmentation_shape=[256, 256, 256], exported_models=len(models),
        exported_segmentations=len(segmentations), slicer_loader_syntax='checked by compile()',
        slicer_gui_run_verified=False, browser_interaction_run_verified=False,
        units='assumed mm, not independently confirmed')
    compile((OUTPUT / 'open_in_slicer.py').read_text(), str(OUTPUT / 'open_in_slicer.py'), 'exec')
    for scan in (source, target):
        expect_hash(scan['row']['parcellation_path'], scan['source_sha256'])
        expect_hash(scan['row']['mesh_npz'], scan['row']['mesh_npz_sha256'])
    write_json(OUTPUT / 'verification.json', verification)
    paths = sorted(p for p in OUTPUT.rglob('*') if p.is_file() and p.name != 'checksums.sha256')
    (OUTPUT / 'checksums.sha256').write_text(''.join(digest(p) + '  ' + str(p.relative_to(OUTPUT)) + '\n' for p in paths), encoding='utf-8')
    archive = OUTPUT.parent / 'brainode_adni1_3d_review.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as file:
        for path in sorted(p for p in OUTPUT.rglob('*') if p.is_file()):
            file.write(path, str(Path(OUTPUT.name) / path.relative_to(OUTPUT)))
    with zipfile.ZipFile(archive) as file:
        assert file.testzip() is None
        for path in sorted(p for p in OUTPUT.rglob('*') if p.is_file()):
            saved = file.read(str(Path(OUTPUT.name) / path.relative_to(OUTPUT)))
            assert hashlib.sha256(saved).hexdigest() == digest(path)
    archive.with_suffix('.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n')
    print(json.dumps(dict(bundle=str(archive), bytes=archive.stat().st_size,
        models=len(models), masks=len(segmentations), example_errors=errors,
        verification_passed=True), indent=2))


if __name__ == '__main__':
    main()
