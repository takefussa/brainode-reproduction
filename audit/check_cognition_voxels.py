"""Diagnostic voxel comparison; selected examples do not define training data."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from cognition import PCAShapeVoxelizer
from lab_shapes import ShapePCA


class DirectVertices:
    def inverse_transform(self,vertices):
        return vertices


def stats(values):
    return dict(min=float(np.min(values)),mean=float(np.mean(values)),max=float(np.max(values)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--method',choices=('surface-fill','center-occupancy'),default='surface-fill')
    args=parser.parse_args()
    root,report=args.data_root.resolve(),args.report.resolve()
    report.mkdir(parents=True,exist_ok=False)
    with np.load(root/'observations.npz',allow_pickle=False) as file:
        data={k:file[k] for k in file.files}
    pca=ShapePCA.load(root/'pca.npz')
    reconstructed=pca.inverse_transform(data['coefficients']).reshape(data['vertices'].shape)
    train=np.flatnonzero(data['splits']=='train')
    augmented=np.concatenate([reconstructed[train]*.98,reconstructed[train]*1.02])
    lower,upper=augmented.min((0,1)),augmented.max((0,1))
    origin=np.floor((lower+upper)/2-31.5)
    with (PROJECT/'artifacts/model_data/adni1_pca150_v1/geometry/geometry.csv').open() as file:
        geometry=list(csv.DictReader(file))
    worst=sorted((r for r in geometry if r['split']=='train'),key=lambda r:float(r['affected_surface_area_percent']),reverse=True)[:8]
    selected={int(r['index']) for r in worst}
    rng=np.random.default_rng(42)
    for label in (0.,1.):
        pool=train[data['conditions'][train]==label]
        selected.update(int(i) for i in rng.choice(pool,size=4,replace=False))
    for axis in range(3):
        selected.add(int(train[np.argmin(reconstructed[train,:,axis].min(1))]))
        selected.add(int(train[np.argmax(reconstructed[train,:,axis].max(1))]))
    selected=sorted(selected)
    records=[];pictures=[]
    for pitch in (1.,.5):
        shape=(int(64/pitch),)*3
        original_voxelizer=PCAShapeVoxelizer(DirectVertices(),data['faces'],grid_shape=shape,origin_mm=origin,pitch_mm=pitch,method=args.method)
        pca_voxelizer=PCAShapeVoxelizer(pca,data['faces'],grid_shape=shape,origin_mm=origin,pitch_mm=pitch,method=args.method)
        alternate_voxelizer=PCAShapeVoxelizer(pca,data['faces'],grid_shape=shape,origin_mm=origin,pitch_mm=pitch,method=args.method,occupancy_rays=7)
        for index in selected:
            original=original_voxelizer(data['vertices'][index].ravel())[0,0].astype(bool)
            compressed=pca_voxelizer(data['coefficients'][index])[0,0].astype(bool)
            alternate=alternate_voxelizer(data['coefficients'][index])[0,0].astype(bool) if args.method=='center-occupancy' else compressed
            a,b=int(original.sum()),int(compressed.sum())
            volume=float(trimesh.Trimesh(vertices=data['vertices'][index],faces=data['faces'],process=False).volume)
            reconstructed_volume=float(trimesh.Trimesh(vertices=reconstructed[index],faces=data['faces'],process=False).volume)
            record=dict(pid=str(data['pids'][index]),uid=str(data['uids'][index]),pitch_mm=pitch,
                original_volume_mm3=volume,pca_volume_mm3=reconstructed_volume,
                original_filled_voxels=a,pca_filled_voxels=b,
                original_voxel_volume_bias_percent=100*(a*pitch**3/volume-1),
                pca_voxel_volume_bias_percent=100*(b*pitch**3/reconstructed_volume-1),
                voxel_volume_change_percent=100*(b/a-1),
                occupancy_ray_vote_disagreement_voxels=int(np.count_nonzero(compressed!=alternate)),
                dice_between_original_and_pca_voxels=float(2*np.count_nonzero(original&compressed)/(a+b)))
            records.append(record)
            if index==int(worst[0]['index']):
                pictures.append((pitch,original,compressed))
            print(json.dumps(record),flush=True)
    with (report/'examples.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary=dict(training_only_grid_fit=True,grid_origin_mm=origin.tolist(),physical_grid_size_mm=[64,64,64],
        train_augmented_bounds=[lower.tolist(),upper.tolist()],
        diagnostic_examples=len(selected),selection='8 train surfaces with highest intersection area, 4 random CN and 4 random AD scans, train coordinate extrema; deduplicated',
        selection_affects_training_dataset=False,methods=args.method,
        pitches_mm=[1.,.5],grid_shapes=[[64,64,64],[128,128,128]],
        by_pitch={str(pitch):{key:stats([r[key] for r in records if r['pitch_mm']==pitch]) for key in
            ['dice_between_original_and_pca_voxels','voxel_volume_change_percent','original_voxel_volume_bias_percent','pca_voxel_volume_bias_percent','occupancy_ray_vote_disagreement_voxels']} for pitch in (1.,.5)},
        cnn_training_started=False,
        limitation='diagnostic examples only; filled occupancy and voxel volume do not validate nonintersecting surface anatomy')
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    for row,(pitch,original,compressed) in enumerate(pictures):
        for axis in range(3):
            occupied=np.argwhere(original|compressed)
            plane=int(np.median(occupied[:,axis]))
            a,b=np.take(original,plane,axis=axis),np.take(compressed,plane,axis=axis)
            image=np.zeros(a.shape+(3,));image[a&b]=[.2,.6,.9];image[a&~b]=[.9,.2,.2];image[b&~a]=[.9,.8,.2]
            axes[row,axis].imshow(image.transpose(1,0,2),origin='lower')
            axes[row,axis].set_title(f'{pitch} mm voxel, axis {axis}');axes[row,axis].set_axis_off()
    fig.suptitle(f"Train example UID {worst[0]['uid']}: shared occupancy blue, original only red, PCA only yellow")
    fig.savefig(report/'voxel_comparison.png',dpi=170);fig.savefig(report/'voxel_comparison.pdf');plt.close(fig)
    (report/'REPORT.md').write_text('\n'.join(['# 連続c推定用voxel化の診断','',
        f"trainのみの{len(selected)}例で元meshとPCA再構成meshを比較した。診断用の選択であり、学習対象の絞込みではない。",'',
        'trainの拡大・縮小後の全PCA形状から固定64mm四方のgridを設定し、1mm/0.5mmの2解像度を比較した。個別の中心合わせ・scale・cropは行わない。', '',
        f"数値: `{json.dumps(summary['by_pitch'])}`。",'',
        f"voxel化方式は `{args.method}`。surface-fillは表面セルのfill、center-occupancyは固定voxel中心のray parity判定を行う。どちらも有限voxelサイズに由来する量子化誤差を持つ。occupancyの比較は自己交差曲面の解剖学的妥当性を証明しない。CNN設定は未確定で、学習を開始していない。", '',
        '[数値とgrid](summary.json)、[全診断例](examples.csv)、[断面比較](voxel_comparison.png)。','']))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
