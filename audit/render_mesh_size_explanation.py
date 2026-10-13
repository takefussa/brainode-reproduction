"""Show template initialization and fitted meshes in the same native mask frame."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'artifacts/mesh_qc/initialization_review'


def native_points(vertices, data):
    inverse = np.linalg.inv(data['icp_matrix'])
    return vertices @ inverse[:3, :3].T + inverse[:3, 3] + data['centroid_header_world']


def main():
    entries = []
    for mode in ('unscaled', 'mask-volume'):
        directory = ROOT / f'ADNI1_34698_{mode}'
        with np.load(directory/'34698.npz') as arrays:
            data = {key: arrays[key].copy() for key in arrays.files}
        record = json.loads((directory/'manifest.json').read_text())['scans'][0]
        target = native_points(data['target'], data)
        entries.append(dict(data=data, record=record, target=target,
                            initial=native_points(data['source_vertices'], data),
                            fitted=native_points(data['vertices'], data)))
    np.testing.assert_allclose(entries[0]['target'], entries[1]['target'], atol=1e-7, rtol=0)
    center = entries[0]['target'].mean(0)
    points = np.vstack([entry[key] for entry in entries for key in ('target', 'initial', 'fitted')]) - center
    lo, hi = points.min(0)-2, points.max(0)+2
    fig = plt.figure(figsize=(12, 9), layout='constrained')
    for row, entry in enumerate(entries):
        for col, key in enumerate(('initial', 'fitted')):
            ax = fig.add_subplot(2, 2, row*2+col+1, projection='3d')
            vertices = entry[key]-center
            faces = entry['data']['faces']
            ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor='#2798d6',
                                                alpha=.45, edgecolor='#267db2', linewidth=.08))
            target = entry['target']-center
            ax.scatter(*target.T, c='#d93c34', s=1.5, alpha=.6, depthshade=False)
            for d, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                setter(lo[d], hi[d])
            ax.set_box_aspect(hi-lo)
            ax.view_init(elev=23, azim=-130)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            if key == 'initial':
                volume = entry['record']['before']['volume_mm3']
                details = f'Template volume {volume:.1f}; observed mask 3036.0'
            else:
                record = entry['record']
                details = f"Volume bias {record['volume_bias_percent']:+.2f}%; Dice {record['overlap']['dice']:.3f}"
            title = 'Original template size' if row == 0 else 'Template size initialized from mask volume'
            ax.set_title(f'{title}\n{key.capitalize()} mesh: {details}', fontsize=10)
    fig.suptitle('ADNI1 / 34698: why initialization changes the fitted result\n'
                 'Red: unchanged observed mask boundary. Blue: initial or fitted template mesh.\n'
                 'Same native mask frame and display scale; units assume 1 numeric unit = 1 mm.', fontsize=12)
    fig.savefig(ROOT/'mesh_size_explanation.png', dpi=150)
    fig.savefig(ROOT/'mesh_size_explanation.pdf')
    plt.close(fig)
    print(ROOT/'mesh_size_explanation.png')


if __name__ == '__main__':
    main()
