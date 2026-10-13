"""Read-only image-grid and foreground-extent audit; no registration or cropping.

All inventory pairs receive header checks. Voxel foreground is measured in the
18 existing cohort examples and the four longitudinal mesh examples. Nonzero
MRI support is a descriptive measurement, not a new brain segmentation.
"""

import csv
import itertools
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / 'artifacts' / 'image_geometry_review'


def bounds_world(affine, low, high):
    corners = np.array(list(itertools.product(*zip(low, high))))
    world = nib.affines.apply_affine(affine, corners)
    return dict(low=world.min(0).tolist(), high=world.max(0).tolist(),
                extent=np.ptp(world, axis=0).tolist())


def header(image):
    shape = np.array(image.shape)
    if shape.shape != (3,):
        raise ValueError(f'Expected a 3D image, got {image.shape}')
    return dict(shape=shape.tolist(), dtype=str(image.get_data_dtype()),
                spacing=[float(x) for x in image.header.get_zooms()],
                orientation=''.join(nib.aff2axcodes(image.affine)),
                units=image.header.get_xyzt_units()[0], affine=image.affine.tolist(),
                fov_world=bounds_world(image.affine, -.5*np.ones(3), shape-.5))


def foreground_info(foreground, affine):
    count = int(foreground.sum())
    if not count:
        return dict(voxels=0, bbox=None)
    low, high = [], []
    for d in range(3):
        occupied = np.flatnonzero(np.any(foreground, axis=tuple(k for k in range(3) if k != d)))
        low.append(int(occupied[0]))
        high.append(int(occupied[-1]))
    dimensions = (np.array(high)-low+1).tolist()
    return dict(voxels=count, bbox=[low, high], bbox_dimensions=dimensions,
                bbox_world=bounds_world(affine, np.array(low)-.5, np.array(high)+.5),
                volume_numeric=count*float(abs(np.linalg.det(affine[:3, :3]))))


def inspect_voxels(path, kind):
    image = nib.load(path)
    values = np.asarray(image.dataobj)
    finite = np.isfinite(values)
    foreground = finite & (values != 0)
    result = dict(path=str(path), **header(image), nonfinite_voxels=int((~finite).sum()),
                  minimum=float(values[finite].min()), maximum=float(values[finite].max()),
                  nonzero_foreground=foreground_info(foreground, image.affine))
    if kind == 'parcellation':
        result['foreground_definition'] = 'finite label != 0; all labelled tissue'
        result['left_hippocampus'] = foreground_info(values == 75, image.affine)
    else:
        result['foreground_definition'] = 'finite MRI intensity != 0; signal support, not an independently validated brain mask'
    return result


def inspect_header_pair(row):
    try:
        parc = nib.load(row['parcellation_path'])
        trsaa = nib.load(row['trsaa_path'])
        a, b = header(parc), header(trsaa)
        return dict(row=row, parcellation=a, trsaa=b,
                    same_shape=a['shape'] == b['shape'],
                    same_affine=bool(np.allclose(parc.affine, trsaa.affine, atol=1e-5, rtol=0)),
                    affine_difference=float(np.max(np.abs(parc.affine-trsaa.affine))))
    except Exception as exc:
        return dict(error=dict(cohort=row['cohort'], uid=row['uid'], error=str(exc)))


def render_example(cohort, uid, record):
    # Native QC MRI is displayed only where the earlier two-file frame audit exists.
    keys = ['unified', 'parcellation', 'trsaa'] if 'unified' in record else ['parcellation', 'trsaa']
    titles = {'unified': 'Native MRI: QC reference only',
              'parcellation': 'Parcellation: mesh source', 'trsaa': 'TRSAA: analysis MRI'}
    fig, axes = plt.subplots(2, len(keys), figsize=(5*len(keys), 9), layout='constrained')
    for col, kind in enumerate(keys):
        entry = record[kind]
        image = nib.load(entry['path'])
        values = np.asarray(image.dataobj)
        lo, hi = np.array(entry['nonzero_foreground']['bbox'])
        center = np.rint((lo+hi)/2).astype(int)
        if kind == 'unified':
            # Same native slice as the parcellation, so these two columns can be compared directly.
            pl, ph = np.array(record['parcellation']['nonzero_foreground']['bbox'])
            center = np.rint((pl+ph)/2).astype(int)
        bright = values[values > 0]
        vmax = float(np.percentile(bright, 99.5)) if kind != 'parcellation' else 280
        for row, (dimension, name) in enumerate([(2, 'Axial'), (1, 'Coronal')]):
            ax = axes[row, col]
            remaining = [d for d in range(3) if d != dimension]
            plane = np.take(values, center[dimension], axis=dimension)
            displayed = np.ma.masked_equal(plane.T, 0) if kind == 'parcellation' else plane.T
            ax.set_facecolor('#eeeeee')
            cmap = plt.get_cmap('turbo' if kind == 'parcellation' else 'gray').copy()
            cmap.set_bad('black')
            ax.imshow(displayed, origin='lower', cmap=cmap,
                      vmin=0, vmax=vmax, extent=(-.5, plane.shape[0]-.5, -.5, plane.shape[1]-.5))
            ax.add_patch(Rectangle((-.5, -.5), plane.shape[0], plane.shape[1],
                                  fill=False, edgecolor='#42bddd', linewidth=1.6))
            a, b = remaining
            ax.add_patch(Rectangle((lo[a]-.5, lo[b]-.5), hi[a]-lo[a]+1, hi[b]-lo[b]+1,
                                  fill=False, edgecolor='#ffc857', linewidth=1.6))
            ax.set_xlim(-.5, 255.5)
            ax.set_ylim(-.5, 255.5)
            ax.set_aspect('equal')
            ax.set_xlabel(f'Local voxel axis {a} (spacing 1)')
            ax.set_ylabel(f'Local voxel axis {b} (spacing 1)')
            dims = ' x '.join(map(str, entry['shape']))
            occupied = ' x '.join(map(str, entry['nonzero_foreground']['bbox_dimensions']))
            ax.set_title(f'{titles[kind]}\nFull grid: {dims}; nonzero box: {occupied}\n{name}, slice {center[dimension]}', fontsize=10)
    fig.suptitle(f'{cohort} / {uid}: full image field and occupied tissue\n'
                 'Equal display scale. Cyan: image field; yellow: 3D nonzero box; gray: outside image.\n'
                 'TRSAA uses its own coordinates and slice center; columns are not registered overlays.', fontsize=12)
    fig.savefig(OUTPUT / f'{cohort}_{uid}_image_sizes.png', dpi=145)
    fig.savefig(OUTPUT / f'{cohort}_{uid}_image_sizes.pdf')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (PROJECT/'artifacts/multicohort_inventory/all_observations.csv').open() as file:
        rows = list(csv.DictReader(file))
    grouped = defaultdict(lambda: dict(pairs=0, parcellation_shapes=Counter(), trsaa_shapes=Counter(),
                                      same_shape=0, same_affine=0, spacings=Counter(), units=Counter()))
    errors = []
    fields = ['cohort', 'uid', 'parcellation_shape', 'trsaa_shape', 'same_shape', 'same_affine',
              'parcellation_spacing', 'trsaa_spacing', 'parcellation_units', 'trsaa_units',
              'affine_max_abs_difference']
    with (OUTPUT/'all_pair_headers.csv').open('w', newline='') as file, ThreadPoolExecutor(max_workers=8) as pool:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, result in enumerate(pool.map(inspect_header_pair, rows)):
            if 'error' in result:
                errors.append(result['error'])
                if len(errors) <= 3:
                    print('Header read error:', result['error'], flush=True)
            else:
                row = result['row']
                a, b = result['parcellation'], result['trsaa']
                same_shape, same_affine = result['same_shape'], result['same_affine']
                group = grouped[row['cohort']]
                group['pairs'] += 1
                group['parcellation_shapes'][str(tuple(a['shape']))] += 1
                group['trsaa_shapes'][str(tuple(b['shape']))] += 1
                group['same_shape'] += int(same_shape)
                group['same_affine'] += int(same_affine)
                group['spacings'][str((tuple(a['spacing']), tuple(b['spacing'])))] += 1
                group['units'][str((a['units'], b['units']))] += 1
                writer.writerow(dict(cohort=row['cohort'], uid=row['uid'],
                    parcellation_shape=str(tuple(a['shape'])), trsaa_shape=str(tuple(b['shape'])),
                    same_shape=same_shape, same_affine=same_affine,
                    parcellation_spacing=str(tuple(a['spacing'])), trsaa_spacing=str(tuple(b['spacing'])),
                    parcellation_units=a['units'], trsaa_units=b['units'],
                    affine_max_abs_difference=result['affine_difference']))
            if (i+1) % 2000 == 0:
                file.flush()
                print('Header pairs checked:', i+1, flush=True)
    samples = json.loads((PROJECT/'artifacts/multicohort_inventory/parcellation_samples.json').read_text())
    selected = {(r['cohort'], r['uid']) for r in samples}
    selected.update(('ADNI1', uid) for uid in ('34698', '62924', '87256', '162792'))
    row_index = {(r['cohort'], r['uid']): r for r in rows}
    records = []
    for cohort, uid in sorted(selected):
        row = row_index[(cohort, uid)]
        record = dict(cohort=cohort, uid=uid)
        record['parcellation'] = inspect_voxels(row['parcellation_path'], 'parcellation')
        record['trsaa'] = inspect_voxels(row['trsaa_path'], 'trsaa')
        if cohort == 'ADNI1' and uid in ('34698', '162792'):
            path = Path('/data2/radiology_datas/clean3/ADNI1/unified') / f'{uid}.nii'
            record['unified'] = inspect_voxels(path, 'unified')
            p, u = record['parcellation'], record['unified']
            record['parcellation_vs_unified_same_grid'] = (
                p['shape'] == u['shape'] and np.allclose(p['affine'], u['affine'], atol=1e-5, rtol=0))
            record['parcellation_vs_unified_same_grid'] = bool(record['parcellation_vs_unified_same_grid'])
        records.append(record)
        print(cohort, uid, 'foreground boxes:', record['parcellation']['nonzero_foreground']['bbox_dimensions'],
              record['trsaa']['nonzero_foreground']['bbox_dimensions'], flush=True)
    summary = dict(date='2026-10-07 Asia/Tokyo', inventory_pairs=len(rows),
                   successful_header_pairs=sum(g['pairs'] for g in grouped.values()),
                   cohorts=grouped, errors=errors, foreground_examples=len(records), records=records,
                   foreground_is_descriptive_not_new_segmentation=True,
                   units_assumption='Header spatial units unknown: numeric spacing 1 interpreted as 1 mm',
                   native_to_trsaa_registration_performed=False, source_files_modified=False)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    for cohort, uid in [('ADNI1', '34698'), ('ADNI3', '1426470')]:
        record = next(r for r in records if (r['cohort'], r['uid']) == (cohort, uid))
        render_example(cohort, uid, record)
    report = ['# 画像格子と脳領域の大きさの比較', '', '2026-10-07 (Asia/Tokyo)。元画像は読み取りのみ。', '',
              f'全{len(rows):,}撮像のparcellation/TRSAA headerを照合し、{summary["successful_header_pairs"]:,}組を確認した。読み取りエラーは{len(errors)}件。', '',
              '| cohort | 確認した組数 | parcellation shape | TRSAA shape | shape一致 | affine一致 |',
              '|---|---:|---|---|---:|---:|']
    for cohort, g in sorted(grouped.items()):
        report.append(f'| {cohort} | {g["pairs"]} | {dict(g["parcellation_shapes"])} | {dict(g["trsaa_shapes"])} | {g["same_shape"]} | {g["same_affine"]} |')
    report += ['', f'voxel内容は{len(records)}例で確認した。以下はmesh比較に使った2例。範囲は各軸の最小〜最大voxelを含むbounding boxで、脳そのものの直径や体積とは異なる。', '',
               '| 例 | データ | 全画像の配列 | 非ゼロ領域の範囲 | 左海馬の範囲 |', '|---|---|---|---|---|']
    for cohort, uid in [('ADNI1', '34698'), ('ADNI3', '1426470')]:
        r = next(r for r in records if (r['cohort'], r['uid']) == (cohort, uid))
        for kind in ('parcellation', 'trsaa'):
            e = r[kind]
            hippo = e.get('left_hippocampus', {}).get('bbox_dimensions', '—')
            report.append(f'| {cohort} {uid} | {kind} | {e["shape"]} | {e["nonzero_foreground"]["bbox_dimensions"]} | {hippo} |')
    report += ['', '非ゼロ領域はparcellationではlabel!=0の組織、MRIではintensity!=0の信号範囲。両者は独立に検証済みの同一brain maskではない。全例の定義・範囲・非有限値・headerは [summary.json](summary.json)、全件headerは [all_pair_headers.csv](all_pair_headers.csv) に記録した。', '',
               '全件のspacing数値は約1/1/1（浮動小数点の保存誤差を含む）で、headerの単位はunknown。サイズをmmで読む場合は1数値単位=1mmという仮定に依存する。', '',
               'ADNI1のUID34698と162792では、parcellationとunifiedの格子・affineが一致する。unifiedはこの2例の座標確認用にのみ読み取った。解析MRIはTRSAAである。TRSAAとの座標変換は確認されておらず、単に配列サイズを揃えるだけでは重ね合わせを保証できない。', '',
               'mesh fittingは既存parcellationのlabel75を使う。境界点をparcellationのaffineで座標へ変換し、中心合わせと拡縮を含まないrigid ICPを実行する。TRSAAの160×224×160配列やそのvoxel indexを入力する処理ではないため、MRIの配列サイズ差とmesh fittingの体積差は別の問題である。', '',
               '初回templateの体積初期化は、初期meshの大きさを調整する論文未記載の補完。画像をTRSAAへ登録する処理ではない。[meshの比較・確認範囲](../mesh_qc/initialization_review/REPORT.md)', '',
               '比較図は各画像自身の座標と断面を使い、表示のvoxel間隔を共通にした。水色は画像全体の枠、黄色は非ゼロ領域の3D bounding boxを断面へ投影した枠。画像内の黒はゼロ背景、画像枠の外側の灰色は表示用の余白である。追加の画像crop・resampling・registrationは実行していない。', '',
               '- [ADNI1の画像サイズ](ADNI1_34698_image_sizes.png)', '- [ADNI3の画像サイズ](ADNI3_1426470_image_sizes.png)', '']
    (OUTPUT/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps({k:summary[k] for k in ('inventory_pairs', 'successful_header_pairs', 'foreground_examples', 'errors')}, indent=2))


if __name__ == '__main__':
    main()
