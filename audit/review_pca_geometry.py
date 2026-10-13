"""Measure residual surface intersections after the fixed PCA150 projection."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import open3d as o3d

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from lab_shapes import ShapePCA


def intersections(vertices,faces):
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
    pairs=np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1,2)
    ids=np.unique(pairs)
    triangles=vertices[faces]
    areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    fraction=float(100*areas[ids].sum()/areas.sum()) if len(ids) else 0.
    return len(pairs),ids,fraction


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    root,report=args.data_root.resolve(),args.report.resolve()
    report.mkdir(parents=True,exist_ok=False)
    with np.load(root/'observations.npz',allow_pickle=False) as file:
        data={k:file[k] for k in file.files}
    pca=ShapePCA.load(root/'pca.npz')
    reconstructed=pca.inverse_transform(data['coefficients']).reshape(data['vertices'].shape)
    faces=data['faces'];records=[];frequency=Counter()
    for index,(original,compressed) in enumerate(zip(data['vertices'],reconstructed)):
        pairs,ids,fraction=intersections(compressed,faces)
        frequency.update(int(i) for i in ids)
        records.append(dict(index=index,pid=str(data['pids'][index]),uid=str(data['uids'][index]),
            split=str(data['splits'][index]),original_quality_flags=str(data['qc_flags'][index]),
            intersection_pairs=pairs,affected_triangles=len(ids),affected_surface_area_percent=fraction))
    count,ids,fraction=intersections(pca.mean_.reshape(-1,3),faces)
    flagged=[r for r in records if r['intersection_pairs']]
    summary=dict(pca_dimension=150,total=len(records),intersecting_scans=len(flagged),
        intersections_by_split=dict(Counter(r['split'] for r in flagged)),
        affected_surface_area_percent_range=[min(r['affected_surface_area_percent'] for r in flagged),max(r['affected_surface_area_percent'] for r in flagged)],
        affected_surface_area_percent_median=float(np.median([r['affected_surface_area_percent'] for r in flagged])),
        affected_surface_area_percent_p95=float(np.quantile([r['affected_surface_area_percent'] for r in flagged],.95)),
        affected_triangles_range=[min(r['affected_triangles'] for r in flagged),max(r['affected_triangles'] for r in flagged)],
        mean_shape_intersection_pairs=count,mean_shape_affected_surface_area_percent=fraction,
        most_frequent_triangle_ids=frequency.most_common(12),
        observations_excluded=0,geometry_repaired=False,
        interpretation='PCA150 does not preserve nonintersecting surfaces; numerical trajectory baseline can be evaluated, while voxel and volume validity require separate verification')
    with (report/'geometry.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    worst=sorted(flagged,key=lambda r:r['affected_surface_area_percent'],reverse=True)[:3]
    fig=plt.figure(figsize=(12,7),layout='constrained')
    for column,r in enumerate(worst):
        index=r['index']
        for row,v in enumerate((data['vertices'][index],reconstructed[index])):
            ax=fig.add_subplot(2,3,row*3+column+1,projection='3d')
            _,ids,_=intersections(v,faces)
            ax.add_collection3d(Poly3DCollection(v[faces],facecolors='#2e87b7',edgecolors='none',alpha=.5))
            if len(ids):
                ax.add_collection3d(Poly3DCollection(v[faces[ids]],facecolors='#d33229',edgecolors='#6f1510',linewidths=.2,alpha=1))
            center=(v.max(0)+v.min(0))/2;span=max(np.ptp(v,axis=0))/2
            ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-60)
            ax.set_title(f"{r['uid']} {'original mesh' if row==0 else 'PCA150'}\n{'' if row==0 else str(round(r['affected_surface_area_percent'],2))+'% affected area'}")
            ax.set_axis_off()
    fig.suptitle('Original and PCA150 surfaces; intersecting triangles shown in red')
    fig.savefig(report/'worst_surface_examples.png',dpi=180);fig.savefig(report/'worst_surface_examples.pdf');plt.close(fig)
    (report/'REPORT.md').write_text('\n'.join(['# PCA150表面geometryの確認','',
        f"1,405撮像中{len(flagged)}撮像で自己交差を検出した。該当面の表面積割合は{summary['affected_surface_area_percent_range']}%、medianは{summary['affected_surface_area_percent_median']}%。",'',
        '固定PCA150の再構成は有限座標と共有faceを保持するが、交差のない曲面を保証しない。元mesh・PCA基底を修復せず、候補撮像を除外しない。', '',
        '観測だけのBrainODE baselineで座標予測の学習・評価は可能。CNN用voxel化と予測体積の妥当性は追加確認が必要で、表面の適合性と座標誤差を同じ結果として扱わない。', '',
        '[元meshと再構成の比較](worst_surface_examples.png)、[全撮像の数値](geometry.csv)、[集計](summary.json)。','']))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
