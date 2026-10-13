"""Standalone scientific figure for the two-case initialization comparison."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'artifacts/mesh_qc/initialization_review'


def main():
    figure=plt.figure(figsize=(12,15))
    reviews=[]
    for row,(cohort,uid) in enumerate([('ADNI1','34698'),('ADNI3','1426470')]):
        entries=[]
        for mode in ('unscaled','mask-volume'):
            directory=ROOT/f'{cohort}_{uid}_{mode}'
            manifest=json.loads((directory/'manifest.json').read_text())
            record=manifest['scans'][0]
            with np.load(directory/f'{uid}.npz') as arrays:
                entries.append((record,{k:arrays[k].copy() for k in ('vertices','faces','target')}))
        union=np.vstack([a[k] for _,a in entries for k in ('vertices','target')])
        center=(union.max(0)+union.min(0))/2
        half=np.ptp(union,axis=0).max()*.53
        for col,(record,data) in enumerate(entries):
            axis=figure.add_subplot(3,2,row*2+col+1,projection='3d')
            axis.add_collection3d(Poly3DCollection(data['vertices'][data['faces']],
                facecolor='#409fda',alpha=.55,linewidth=0))
            axis.scatter(*data['target'].T,color='#da4938',s=.8,alpha=.25)
            for setter,value in zip((axis.set_xlim,axis.set_ylim,axis.set_zlim),center):
                setter(value-half,value+half)
            axis.set_box_aspect((1,1,1));axis.view_init(elev=20,azim=-60)
            axis.set_xlabel('x (assumed mm)');axis.set_ylabel('y');axis.set_zlabel('z')
            mode='Unscaled template' if col==0 else 'Template initialized to mask volume'
            axis.set_title(f'{cohort} / {uid}: {mode}\n'
                f"Volume bias {record['volume_bias_percent']:+.2f}%; Dice {record['overlap']['dice']:.3f}\n"
                f"Self-intersection pairs {record['after']['self_intersecting_triangle_pairs']}",fontsize=10)
        axis=figure.add_subplot(3,2,5+row)
        mask=entries[0][0]['mask_volume_mm3']
        values=[mask,*[r['after']['volume_mm3'] for r,_ in entries]]
        bars=axis.bar(['Original mask','Unscaled fit','Size-initialized fit'],values,
                     color=['#da4938','#8c96a1','#409fda'])
        for bar,v in zip(bars,values):
            axis.text(bar.get_x()+bar.get_width()/2,v+max(values)*.02,f'{v:.1f}',ha='center',fontsize=10)
        axis.set_ylim(0,max(values)*1.16)
        axis.set_ylabel('Volume (assumed mm3)');axis.set_title(f'{cohort} / {uid}')
        reviews.append(dict(cohort=cohort,uid=uid,unscaled=entries[0][0],mask_volume_initialized=entries[1][0]))
    figure.suptitle('Template size affects fitting of the lab hippocampal masks\n'
        'Blue: fitted mesh; red: observed mask boundary. 1490 vertices / 2976 faces preserved.\n'
        'Same seed, PointNet, seven losses, 5000 updates. Initialization changes the template only.',fontsize=13)
    figure.tight_layout(rect=(0,0,1,.93))
    figure.savefig(ROOT/'initialization_comparison.png',dpi=180)
    figure.savefig(ROOT/'initialization_comparison.pdf')
    plt.close(figure)
    (ROOT/'comparison.json').write_text(json.dumps(dict(date='2026-10-07 Asia/Tokyo',
        cases=reviews,scope='two engineering cases, not full-cohort validation',
        modified_initialization_is_a_paper_supplement=True,
        masks_and_final_volumes_not_normalized=True),indent=2))
    print(ROOT/'initialization_comparison.png')


if __name__=='__main__':
    main()
