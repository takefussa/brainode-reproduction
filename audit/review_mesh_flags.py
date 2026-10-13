"""Quantify candidate quality flags without repairing or selecting scans."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import open3d as o3d
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('quality reports must remain under the home project')
    progress = json.loads((root/'progress.json').read_text())
    if progress['status'] != 'completed_candidates':
        parser.error('the completed batch is required for final flag review')
    output.mkdir(parents=True, exist_ok=False)
    records, frequency = [], Counter()
    for path in sorted(root.glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        for scan in manifest['scans']:
            if not scan['qc_flags']:
                continue
            entry = dict(subject=manifest['subject'], uid=scan['uid'], day=scan['day'],
                split=scan['split'], diagnosis=scan['diagnosis'], flags=scan['qc_flags'],
                dice=scan['overlap']['dice'], volume_bias_percent=scan['volume_bias_percent'],
                boundary_mean_mm=scan['after']['boundary_to_surface_mean_mm'])
            if 'self_intersection' in scan['qc_flags']:
                npz = path.parent/f"{scan['uid']}.npz"
                if hashlib.sha256(npz.read_bytes()).hexdigest() != scan['artifact_sha256'][npz.name]:
                    raise ValueError('candidate mesh changed')
                with np.load(npz) as data:
                    vertices, faces = data['vertices'].copy(), data['faces'].copy()
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
                pairs = np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1,2)
                if len(pairs) != scan['after']['self_intersecting_triangle_pairs']:
                    raise ValueError('intersection count differs from saved fitting result')
                ids = np.unique(pairs)
                triangles = vertices[faces]
                areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]),axis=1)/2
                points = triangles[ids].reshape(-1,3)
                entry.update(intersection_pairs=pairs.tolist(), affected_triangle_ids=ids.tolist(),
                    affected_triangles=len(ids), total_triangles=len(faces),
                    affected_surface_area_percent=float(100*areas[ids].sum()/areas.sum()),
                    affected_bbox_extent_mm=np.ptp(points,axis=0).tolist(),
                    affected_triangle_area_min_mm2=float(areas[ids].min()),
                    affected_triangle_area_max_mm2=float(areas[ids].max()))
                frequency.update(int(i) for i in ids)
            if 'mask_multiple_components' in scan['qc_flags']:
                source = Path(scan['source_path'])
                if hashlib.sha256(source.read_bytes()).hexdigest() != scan['source_sha256']:
                    raise ValueError('original parcellation changed')
                mask = np.asarray(nib.load(source).dataobj)==75
                components, count = ndimage.label(mask,structure=np.ones((3,3,3),dtype=np.uint8))
                sizes = sorted(np.bincount(components.ravel())[1:].tolist(),reverse=True)
                entry.update(mask_components=int(count), component_voxels=sizes,
                    secondary_component_voxels=sum(sizes[1:]),
                    secondary_component_percent=float(100*sum(sizes[1:])/sum(sizes)))
            records.append(entry)
    intersections = [r for r in records if 'self_intersection' in r['flags']]
    components = [r for r in records if 'mask_multiple_components' in r['flags']]
    low_dice = [r for r in records if 'low_or_unavailable_dice' in r['flags']]
    summary = dict(scans=progress['completed_scans'], flagged_scans=len(records),
        flags=dict(Counter(f for r in records for f in r['flags'])),
        intersection_pair_range=[min(len(r['intersection_pairs']) for r in intersections),max(len(r['intersection_pairs']) for r in intersections)],
        affected_triangles_range=[min(r['affected_triangles'] for r in intersections),max(r['affected_triangles'] for r in intersections)],
        affected_surface_area_percent_range=[min(r['affected_surface_area_percent'] for r in intersections),max(r['affected_surface_area_percent'] for r in intersections)],
        secondary_component_voxels_range=[min(r['secondary_component_voxels'] for r in components),max(r['secondary_component_voxels'] for r in components)],
        secondary_component_percent_range=[min(r['secondary_component_percent'] for r in components),max(r['secondary_component_percent'] for r in components)],
        low_dice_range=[min(r['dice'] for r in low_dice),max(r['dice'] for r in low_dice)],
        frequent_intersecting_triangle_ids=frequency.most_common(20),
        source_and_candidate_modified=False, observations_excluded=0,
        geometry_review='all finite shared-topology candidates retained; flagged geometry remains present',
        future_prediction_performance='not evaluated')
    (output/'details.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=[dict(subject=r['subject'],uid=r['uid'],split=r['split'],diagnosis=r['diagnosis'],
        flags=';'.join(r['flags']),dice=r['dice'],volume_bias_percent=r['volume_bias_percent'],
        intersection_pairs=len(r.get('intersection_pairs',[])),affected_triangles=r.get('affected_triangles',0),
        affected_surface_area_percent=r.get('affected_surface_area_percent',0),
        component_voxels=';'.join(str(v) for v in r.get('component_voxels',[])),
        secondary_component_percent=r.get('secondary_component_percent',0)) for r in records]
    with (output/'flag_details.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=['# ADNI1 meshの品質flag詳細','',
        f"全{summary['scans']}撮像を保持し、{len(records)}撮像のflagを確認した。修復・除外・再fittingは行っていない。",'',
        f"自己交差{len(intersections)}撮像: 各{summary['intersection_pair_range']}組、関係する面は{summary['affected_triangles_range']}枚/2976面。関係する面の表面積割合は{summary['affected_surface_area_percent_range']}%。",'',
        f"mask分離{len(components)}撮像: 主成分以外の合計voxelは{summary['secondary_component_voxels_range']}、元maskに対する割合は{summary['secondary_component_percent_range']}%。",'',
        f"Dice<0.90の{len(low_dice)}撮像は範囲{summary['low_dice_range']}。",'',
        '共有faceと有限頂点、保存物の整合性は全件確認済み。自己交差は局所の問題であっても残存するため、将来予測の評価ではflag別の数値も示す。表面の完全な解剖学的妥当性を証明した結果とは扱わない。', '',
        '[全flagとgeometry](details.json)、[CSV](flag_details.csv)、[集計](summary.json)。','']
    (output/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
