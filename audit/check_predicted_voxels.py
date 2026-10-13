"""Check fixed-grid voxelization of every saved observed-baseline forecast."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from cognition import PCAShapeVoxelizer


class DirectVertices:
    def inverse_transform(self,coefficients):
        return coefficients


def main():
    run=PROJECT/'artifacts/training_runs/adni1_observed_v1'
    progress=json.loads((run/'progress.json').read_text())
    path=Path(progress['output'])/'future_predictions.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=progress['evaluation']['prediction_file_sha256']:
        raise ValueError('saved predictions changed')
    grid_config=json.loads((PROJECT/'artifacts/cognition_voxels/adni1_pca150_center05_v1/config.json').read_text())
    with np.load(path,allow_pickle=False) as file:
        predictions={k:file[k].copy() for k in file.files}
    with (run/'future_metrics.csv').open() as file:
        metrics=list(csv.DictReader(file))
    def check(index):
        vertices=predictions['vertices'][index]
        record=dict(index=index,pid=str(predictions['pids'][index]),target_uid=str(predictions['target_uids'][index]),
            self_intersection_pairs=int(metrics[index]['prediction_self_intersection_pairs']))
        try:
            settings=dict(grid_shape=grid_config['grid_shape'],origin_mm=grid_config['origin_mm'],
                          pitch_mm=grid_config['pitch_mm'],method='center-occupancy')
            first=PCAShapeVoxelizer(DirectVertices(),predictions['faces'],**settings,occupancy_rays=3)(vertices.ravel())
            second=PCAShapeVoxelizer(DirectVertices(),predictions['faces'],**settings,occupancy_rays=7)(vertices.ravel())
            mesh=trimesh.Trimesh(vertices=vertices,faces=predictions['faces'],process=False)
            volume=float(mesh.volume)
            if volume<=0 or not np.isfinite(volume):
                raise ValueError('nonpositive or nonfinite signed mesh volume')
            record.update(status='voxelized',occupied_voxels=int(first.sum()),
                ray_vote_disagreement_voxels=int(np.count_nonzero(first!=second)),
                signed_mesh_volume_mm3=volume,
                voxel_to_signed_mesh_volume_bias_percent=float(100*(first.sum()*.125/volume-1)))
        except Exception as error:
            record.update(status='failed',error=repr(error))
        return record
    with ThreadPoolExecutor(max_workers=4) as executor:
        records=list(executor.map(check,range(len(metrics))))
    successes=[r for r in records if r['status']=='voxelized']
    failures=[r for r in records if r['status']=='failed']
    summary=dict(predictions=len(records),voxelized=len(successes),failed=len(failures),failures=failures,
        self_intersecting_predictions=sum(r['self_intersection_pairs']>0 for r in records),
        scans_with_ray_vote_disagreement=sum(r['ray_vote_disagreement_voxels']>0 for r in successes),
        max_abs_voxel_to_signed_mesh_volume_bias_percent=max(abs(r['voxel_to_signed_mesh_volume_bias_percent']) for r in successes),
        grid=dict(shape=grid_config['grid_shape'],pitch_mm=grid_config['pitch_mm'],origin_mm=grid_config['origin_mm']),
        predicted_mesh_modified=False,observations_excluded=0,cnn_training_started=False,
        scope='saved endpoint predictions only; not pseudo samples or all Algorithm 2 intermediate states',
        limitation='ray agreement and small discrepancy from signed mesh volume do not establish valid anatomical interiors or repair self-intersection',
        method_source='https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html#open3d.t.geometry.RaycastingScene.compute_occupancy')
    report=run/'predicted_voxel_review';report.mkdir(exist_ok=False)
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (report/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    lines=['# 初期BrainODE予測のvoxel化確認','',
        f"全{len(records)}予測のうち{len(successes)}件が既存の固定gridでvoxel化でき、失敗は{len(failures)}件。crop・拡縮・修復・対象除外は行っていない。",'',
        f"自己交差ありは{summary['self_intersecting_predictions']}件。3/7 raysでvoxel判定が異なる形状は{summary['scans_with_ray_vote_disagreement']}件、voxel体積と符号付きmesh体積の差は最大{summary['max_abs_voxel_to_signed_mesh_volume_bias_percent']:.4f}%。",'',
        'この結果は保存した予測endpointでの数値的整合性を示す。自己交差の解消・正しい解剖学的内部・CNNのc出力への影響がないことを示さない。符号付きmesh体積自体も、自己交差する曲面では実際の内部体積と同じ意味とは限らない。', '',
        'pseudo形状とAlgorithm 2途中の形状は未検証。CNN学習前に同じ確認を行い、既知のgeometry flagを保持する。', '',
        f"[全数値](summary.json)、[撮像ごとの結果](records.json)、[内部判定の前提]({summary['method_source']})。",'']
    (report/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
