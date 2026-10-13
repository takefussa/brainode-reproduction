"""Export observed meshes in the native parcellation RAS frame for Slicer."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--subject-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.subject_directory.resolve(), args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('light visualization exports must remain under the home project')
    manifest = json.loads((directory/'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for record in manifest['scans']:
        source = Path(record['source_path'])
        if sha(source) != record['source_sha256']:
            raise ValueError('original parcellation changed')
        artifact = directory/f"{record['uid']}.npz"
        if sha(artifact) != record['artifact_sha256'][artifact.name]:
            raise ValueError('saved mesh artifact changed')
        with np.load(artifact) as data:
            fitted, faces = data['vertices'].copy(), data['faces'].copy()
            transform = data['icp_matrix'].copy()
            centroid, affine = data['centroid_header_world'].copy(), data['affine'].copy()
        inverse = np.linalg.inv(transform)
        centered = fitted @ inverse[:3,:3].T + inverse[:3,3]
        native = centered + centroid
        np.testing.assert_allclose(centered @ transform[:3,:3].T + transform[:3,3], fitted, atol=1e-10, rtol=0)
        prefix = f"{record['day']}_{record['diagnosis']}_{record['uid']}"
        obj, vtk = output/f'{prefix}_native_RAS.obj', output/f'{prefix}_native_RAS.vtk'
        with obj.open('w') as file:
            file.write('# Observed hippocampal mesh in native parcellation coordinates. SPACE=RAS\n')
            for vertex in native:
                file.write('v '+' '.join(f'{value:.17g}' for value in vertex)+'\n')
            for face in faces:
                file.write('f '+' '.join(str(int(value)+1) for value in face)+'\n')
        with vtk.open('w') as file:
            file.write('# vtk DataFile Version 3.0\nNative parcellation mesh. SPACE=RAS\nASCII\nDATASET POLYDATA\n')
            file.write(f'POINTS {len(native)} double\n')
            for vertex in native:
                file.write(' '.join(f'{value:.17g}' for value in vertex)+'\n')
            file.write(f'POLYGONS {len(faces)} {4*len(faces)}\n')
            for face in faces:
                file.write('3 '+' '.join(str(int(value)) for value in face)+'\n')
        restored = trimesh.load_mesh(obj, process=False)
        np.testing.assert_array_equal(restored.faces, faces)
        np.testing.assert_allclose(restored.vertices, native, atol=1e-10, rtol=0)
        source_volume = abs(trimesh.Trimesh(fitted, faces, process=False).volume)
        np.testing.assert_allclose(abs(restored.volume), source_volume, atol=1e-8, rtol=1e-10)
        image = nib.load(source)
        np.testing.assert_array_equal(image.affine, affine)
        mask = (np.asarray(image.dataobj)==75).astype(np.uint8)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_slope_inter(1, 0)
        header.set_intent('none')
        header['descrip'] = b'Binary Hippo_L mask extracted from source label75; unchanged native grid'
        mask_path = output/f'{prefix}_HippoL.nii.gz'
        nib.save(nib.Nifti1Image(mask, image.affine, header=header), mask_path)
        reloaded = nib.load(mask_path)
        np.testing.assert_array_equal(reloaded.affine, image.affine)
        np.testing.assert_array_equal(np.asarray(reloaded.dataobj), mask)
        if sha(source) != record['source_sha256'] or sha(artifact) != record['artifact_sha256'][artifact.name]:
            raise ValueError('original source or mesh changed during export')
        records.append(dict(uid=record['uid'], day=record['day'], diagnosis=record['diagnosis'],
            vtk=vtk.name, obj=obj.name, mask=mask_path.name, original_parcellation=str(source),
            original_parcellation_sha256=record['source_sha256'], original_mesh_npz=str(artifact),
            original_mesh_sha256=record['artifact_sha256'][artifact.name],
            coordinate_system='native parcellation RAS, explicitly declared SPACE=RAS in model header',
            vertices=len(native), faces=len(faces), mask_shape=list(mask.shape), mask_voxels=int(mask.sum()),
            native_transform_roundtrip_verified=True, mesh_volume_preserved=True,
            native_affine_and_mask_exact=True,
            export_sha256={p.name:sha(p) for p in (obj,vtk,mask_path)}))
    result = dict(subject=manifest['subject'], use='Slicer visualization only; no training artifacts changed',
        transform='fitted coordinates -> inverse ICP -> restore original boundary centroid',
        source_units='unknown; numeric coordinates interpreted as mm without rescaling',
        mask='source label75 -> binary foreground1; no resampling or component removal',
        reference_grid='native parcellation; TRSAA overlay transform remains unverified', scans=records)
    (output/'export_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    docs=['# Slicer用の左海馬meshと3Dマスク','',
        '各撮像のmeshを元parcellationのRAS座標へ戻し、model headerへSPACE=RASを明示した。元mask、学習用mesh、元画像は変更していない。', '',
        'SlicerのFile → Add Dataで、.vtk（または.obj）をModel、同じ日付・UIDの_HippoL.nii.gzをSegmentationとして読み込む。複数時点を同時表示すると重なるため、まず1時点ずつ表示する。', '',
        'NIfTIは左海馬label75のみをforeground1として取り出した3D volume。元parcellationと同じ256³格子とaffineを保持し、resamplingやmask成分の削除は行っていない。', '',
        'このmeshはnative parcellationとの重ね合わせ用。TRSAA画像とのregistrationは未確認。座標の数値1を1mmとする既存仮定を保ち、拡縮を行わない。', '',
        '[出力と検証の全記録](export_manifest.json)。', '',
        '形式・座標規約: [Slicer公式ドキュメント](https://slicer.readthedocs.io/en/latest/user_guide/data_loading_and_saving.html#specifying-the-coordinate-system-in-model-files)。','']
    (output/'README.md').write_text('\n'.join(docs))
    print(json.dumps({k:v for k,v in result.items() if k!='scans'},indent=2))
    print('Verified native-coordinate meshes and exact binary masks:',len(records))


if __name__ == '__main__':
    main()
