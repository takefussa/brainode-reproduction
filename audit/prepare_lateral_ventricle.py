"""Audit public LV assets and export observed ventricles without prediction.

Derived template files are separate from untouched third-party sources. The
preview contains exact voxel-cell surfaces, not learned correspondence meshes.
"""

import csv
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import zipfile

import nibabel as nib
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy import ndimage
import trimesh

from export_review_bundle import digest, voxel_boundary, write_mesh, write_json

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'references/LV_Shape_Modeling'
AUDIT = PROJECT / 'artifacts/lateral_ventricle/preparation_v1'
PREVIEW = PROJECT / 'artifacts/review_bundles/adni1_LV_RID352_observed'
SIDES = {'left': [163, 164, 165, 166, 167], 'right': [168, 169, 170, 171, 172]}


class NumpyArraysOnly(pickle.Unpickler):
    def find_class(self, module, name):
        permitted = {
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
        }
        if (module, name) not in permitted:
            raise ValueError('Unsupported template pickle object: ' + str((module, name)))
        return permitted[(module, name)]


def prepare_template():
    path = SOURCE / 'temp_meshes/cuttail_temp_L_tex.pkl'
    before_hash = digest(path)
    with path.open('rb') as file:
        original = NumpyArraysOnly(file).load()
    vertices = np.asarray(original['cuttail_vert'], dtype=np.float64)
    faces = np.asarray(original['cuttail_lv'], dtype=np.int64)
    raw = trimesh.Trimesh(vertices, faces, process=False)
    # Preserve the first occurrence and order of each nondegenerate face.
    _, unique_indices = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    selected = np.sort(unique_indices)
    selected = selected[raw.area_faces[selected] > 1e-12]
    retained = faces[selected]
    indices = np.unique(retained)
    reindex = np.full(len(vertices), -1, dtype=np.int64)
    reindex[indices] = np.arange(len(indices))
    clean_faces = reindex[retained]
    clean_vertices = vertices[indices]
    clean = trimesh.Trimesh(clean_vertices, clean_faces, process=False)
    if not clean.is_watertight or not clean.is_winding_consistent or clean.euler_number != 2:
        raise ValueError('Public LV topology still invalid after removing redundant faces')
    np.testing.assert_allclose(clean.volume, raw.volume, atol=1e-8, rtol=1e-12)
    # The author's extraction code ignores header affine and uses LAS-indexed
    # SynthSeg examples. Convert axis orientation before fitting RAS-world masks.
    example = nib.load(SOURCE / 'test_data/test1/synthseg.nii.gz')
    if nib.aff2axcodes(example.affine) != ('L', 'A', 'S'):
        raise ValueError('Public example orientation changed')
    centroid = clean_vertices.mean(axis=0)
    assets = {}
    for side, x_sign in [('left', -1), ('right', 1)]:
        derived = (clean_vertices - centroid) * np.array([x_sign, 1, 1])
        derived_faces = clean_faces.copy()
        mesh = trimesh.Trimesh(derived, derived_faces, process=False)
        if mesh.volume < 0:
            derived_faces = derived_faces[:, [0, 2, 1]]
            mesh = trimesh.Trimesh(derived, derived_faces, process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        surface = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(derived),
                                           o3d.utility.Vector3iVector(derived_faces))
        pairs = np.asarray(surface.get_self_intersecting_triangles())
        if len(pairs):
            raise ValueError('Derived LV template contains self-intersections')
        prefix = AUDIT / f'lv_{side}_template'
        vtk, obj = write_mesh(prefix, derived, derived_faces)
        numeric = prefix.with_suffix('.npz')
        np.savez_compressed(numeric, vertices=derived, faces=derived_faces,
            original_vertex_indices=indices, original_face_indices=selected)
        assets[side] = dict(npz=str(numeric), vtk=str(vtk), obj=str(obj),
            sha256={p.name: digest(p) for p in (numeric, vtk, obj)},
            vertices=len(derived), faces=len(derived_faces), volume=mesh.volume,
            self_intersection_pairs=0, watertight=True, winding_consistent=True,
            labels=SIDES[side], centroid_subtracted=centroid.tolist(),
            x_sign_from_original_template=x_sign)
    assert digest(path) == before_hash
    report = dict(public_repository='https://github.com/PWonjung/LV_Shape_Modeling',
        public_commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        source_template=str(path), source_template_sha256=before_hash,
        original=dict(vertices=len(vertices), lv_referenced_vertices=len(np.unique(faces)), faces=len(faces),
            duplicate_faces=int(len(faces)-len(unique_indices)), zero_area_faces=int((raw.area_faces <= 1e-12).sum()),
            watertight=bool(raw.is_watertight), signed_volume=raw.volume),
        derived_policy='Remove duplicate and zero-area LV faces, remove unused vertices, preserve retained vertex/face order, center once, convert LAS-index axes to RAS, orient outward; mirror for right',
        removed_faces=int(len(faces)-len(clean_faces)), removed_volume_abs=float(abs(clean.volume-raw.volume)),
        voxel_resampling=False, anatomical_smoothing=False, holes_filled=False,
        orientation=dict(reference_example=str(SOURCE/'test_data/test1/synthseg.nii.gz'),
            reference_example_sha256=digest(SOURCE/'test_data/test1/synthseg.nii.gz'),
            axis_codes=list(nib.aff2axcodes(example.affine)),
            interpretation='LAS-index template convention inferred from the released extraction code and examples; the complete historical template-generation affine is not available',
            caveat='Provided template NIfTI has identity affine; it is not evidence of a verified native MRI mapping'),
        assets=assets, original_sources_unchanged=True)
    write_json(AUDIT / 'template_audit.json', report)
    return report


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    data_dir = PREVIEW / 'data'
    data_dir.mkdir(exist_ok=True)
    report = prepare_template()
    eligible_path = PROJECT / 'artifacts/adni1_stage_baseline/eligible.csv'
    with eligible_path.open() as file:
        eligible = list(csv.DictReader(file))
    if len(eligible) != 1405 or len({r['pid'] for r in eligible}) != 468:
        raise ValueError('ADNI1 cohort changed')
    write_json(AUDIT / 'cohort_contract.json', dict(
        cohort='ADNI1', patients=468, scans=1405, eligible_csv=str(eligible_path),
        eligible_csv_sha256=digest(eligible_path),
        train_patients=389, train_scans=1150, test_patients=79, test_scans=255,
        time='(age-55.7288)/37.2931; frozen train-fitted scale',
        age='all finite CN/AD scan ages; no age-range restriction',
        diagnosis='exclude MCI scans; retain eligible scans of patients with MCI history',
        label_definition_candidates=SIDES,
        separate_choroid_labels={'left':272,'right':273},
        region_scope_status='pending left-only versus separate left/right model choice',
        brainode_training_started=False, future_predictions_available=False))
    scans, model_records, arrays = [], [], []
    for date_key, uid in [('baseline', '34537'), ('followup', '176917')]:
        row = next(r for r in eligible if r['uid'] == uid)
        path = Path(row['parcellation_path'])
        before_hash = digest(path)
        image = nib.load(path)
        source = np.asarray(image.dataobj)
        bilateral = np.zeros(source.shape, dtype=np.uint8)
        for side, labels in SIDES.items():
            mask = np.isin(source, labels)
            bilateral[mask] = 1 if side == 'left' else 2
            indices = np.argwhere(mask)
            components, count = ndimage.label(mask, structure=np.ones((3,3,3), dtype=np.uint8))
            sizes = np.sort(np.bincount(components.ravel())[1:])[::-1]
            prefix = f'{date_key}_{side}_LV'
            mask_path = data_dir / f'{prefix}_segmentation.nii.gz'
            header = image.header.copy()
            header.set_data_dtype(np.uint8)
            header.set_slope_inter(1, 0)
            header.set_intent('none')
            header.set_xyzt_units('mm', 'unknown')
            header['descrip'] = b'Observed LV mask; assumed mm; native grid; no resampling'
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine, header=header), mask_path)
            restored = nib.load(mask_path)
            np.testing.assert_array_equal(np.asarray(restored.dataobj), mask)
            np.testing.assert_array_equal(restored.affine, image.affine)
            v, f = voxel_boundary(mask, image.affine)
            vtk, obj = write_mesh(data_dir / f'{prefix}_observed_voxel_surface', v, f)
            color = [0.18, 0.46, 0.91] if side == 'left' else [0.88, 0.34, 0.21]
            model_records.append(dict(name=prefix, date_key=date_key, side=side,
                vtk=str(vtk.relative_to(PREVIEW)), obj=str(obj.relative_to(PREVIEW)),
                mask=str(mask_path.relative_to(PREVIEW)), color=color,
                interpretation='Exact observed voxel boundary; no PointNet fitting or future prediction',
                vertices=len(v), faces=len(f)))
            arrays.append((v, f))
            scans.append(dict(uid=uid, day=row['day'], pid=row['pid'], diagnosis=row['class'],
                side=side, labels=labels, mask_voxels=int(mask.sum()),
                volume_assumed_mm3=float(mask.sum()*abs(np.linalg.det(image.affine[:3,:3]))),
                mask_shape=list(mask.shape), mask_bbox_shape=(indices.max(0)-indices.min(0)+1).tolist(),
                components_26=int(count), component_sizes=sizes.tolist(),
                native_affine=image.affine.tolist(), original_spatial_units=image.header.get_xyzt_units()[0],
                original_source=str(path), original_source_sha256=before_hash,
                original_voxel_values_preserved=True, resampling=False,
                excluded_choroid_label=272 if side=='left' else 273,
                choroid_voxels=int((source==(272 if side=='left' else 273)).sum()),
                components_removed=False))
        bilateral_path = data_dir / f'{date_key}_bilateral_LV_segmentation.nii.gz'
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_intent('none')
        header.set_xyzt_units('mm', 'unknown')
        nib.save(nib.Nifti1Image(bilateral, image.affine, header=header), bilateral_path)
        restored = nib.load(bilateral_path)
        np.testing.assert_array_equal(restored.affine, image.affine)
        np.testing.assert_array_equal(np.asarray(restored.dataobj), bilateral)
        assert digest(path) == before_hash
    manifest = dict(pid='ADNI:RID:352', models=model_records, scans=scans,
        source_day='2006-04-07', followup_day='2010-04-28',
        coordinate_system='Each scan retains its own native parcellation RAS frame; no longitudinal alignment inferred',
        units='assumed mm; original header spatial units unknown',
        forecast_available=False, fitted_correspondence_mesh_available=False,
        surface_method='Exact exposed voxel-cell faces; not the learned BrainODE input mesh',
        label_definition='LV cavity labels163..167 left and168..172 right; choroid272/273 separate and not added; no third/fourth ventricles',
        segmentation_components='all preserved', verification=dict(mask_values_exact=True,
            affine_exact=True, vtk_obj_readback=True, voxel_surface_volume_exact=True,
            original_sources_unchanged=True, slicer_gui_run=False))
    write_json(PREVIEW / 'preview_manifest.json', manifest)
    write_json(PREVIEW / 'template_audit.json', report)
    fig = go.Figure()
    for record, (v,f) in zip(model_records, arrays):
        color = 'rgb('+','.join(str(round(c*255)) for c in record['color'])+')'
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            color=color,name=record['name'],opacity=0.75,showlegend=True,
            visible=record['date_key']=='baseline'))
    bounds=np.concatenate([v for v,f in arrays]);lo,hi=bounds.min(0)-3,bounds.max(0)+3
    buttons=[dict(label=title,method='update',args=[{'visible':[r['date_key']==key for r in model_records]},
             {'title':title+' — 実測segmentationの3D表面'}])
             for key,title in [('baseline','初回 2006-04-07'),('followup','実測4年後 2010-04-28')]]
    fig.update_layout(title='初回 2006-04-07 — 実測segmentationの3D表面',height=740,
        scene=dict(aspectmode='data',xaxis=dict(title='R（仮定mm）',range=[lo[0],hi[0]]),
                   yaxis=dict(title='A（仮定mm）',range=[lo[1],hi[1]]),
                   zaxis=dict(title='S（仮定mm）',range=[lo[2],hi[2]])),
        updatemenus=[dict(buttons=buttons,x=0,y=1.12)],margin=dict(t=100,l=10,r=10,b=10))
    intro=('<h1>側脳室・実測形状</h1><p>RID352。青＝左側脳室、赤＝右側脳室。'
        '回転・拡大可能。メニューで初回と実測4年後を切り替える。'
        '各時点のnative RASを保持し、長期registrationは行っていない。</p>'
        '<p>これはsegmentationのvoxel表面であり、PointNetの対応付きmeshやBrainODEの将来予測ではない。'
        '側脳室の将来予測モデルは未学習。</p>')
    (PREVIEW/'viewer.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8">'
        '<title>側脳室 3D</title><body style="font-family:sans-serif">'+intro+
        fig.to_html(full_html=False,include_plotlyjs=True)+'</body></html>',encoding='utf-8')
    docs='''# ADNI1 側脳室・実測segmentationの3D確認

患者RID352の2006-04-07と2010-04-28を使用した。左海馬の既存previewと同じ患者・日付である。

## ファイル

- `data/baseline_left_LV_segmentation.nii.gz`：初回の左側脳室。
- `data/baseline_right_LV_segmentation.nii.gz`：初回の右側脳室。
- `data/baseline_bilateral_LV_segmentation.nii.gz`：左右を同じvolumeで表示。左1、右2、背景0。
- `data/baseline_left_LV_observed_voxel_surface.vtk`：左maskのvoxel境界をそのまま3D化。
- `data/baseline_right_LV_observed_voxel_surface.vtk`：右maskのvoxel境界をそのまま3D化。
- `followup_*`：4年後の実測データ。同じ種類のNIfTI・VTK・OBJ。
- `viewer.html`：オフラインの回転・拡大・時点切替。

NIfTIはSlicerのFile → Add DataでSegmentationとして読み込む。立体表示はShow 3Dで有効にする。VTKはModelとして読み込めば直接立体表示できる。headerにSPACE=RASを明示した。OBJは同じgeometryの汎用形式。

## 抽出範囲・座標

左：既存atlasの163 LV_Frontal_L、164 LV_body_L、165 LV_atrium_L、166 LV_Occipital_L、167 LV_Inferior_Lを合成。右：168〜172の対応領域を合成。Chroid_LVetc_L/R=272/273、III/IV ventricleは今回のcavity表示には加えていない。5領域内の全voxel・小成分を保持した。maskの成分を削除せず、補間・平滑化・closingを行っていない。この抽出定義は学習済みdatasetの変更ではなく、側脳室用設定を確認するための表示出力である。

各撮像の256³格子とnative affineを完全保持した。初回のbboxは左34×99×48、右37×102×47 voxels。海馬より長く、海馬用128³・pitch0.5の64mm gridは側脳室にはそのまま使えない。将来用CNN gridは側脳室train meshから別途設定・検証する。

元画像の空間単位はunknown。数値1=1mmという既存の仮定を維持し、出力の単位のみ表示用にmmと明示した。時点間registrationは行っていないので、初回とfollowupを重ねた差をそのまま形状予測誤差としない。

## 学習・予測の状態

このパッケージのsurfaceは、foreground voxelの露出面を三角形で表示するルールベースの表面である。対応付きPointNet meshでも、BrainODEによる予測でもない。4年後のファイルは実際に観測された形状であり、予測として扱わない。

側脳室用には別のPCA basis、BrainODE checkpoint、CNN estimatorとvoxel gridが必要。海馬のモデル・PCAを流用して予測やスコアを作らない。ADNI1の対象撮像・患者分割・年齢/timeは固定cohortのまま再利用する。左右を別々に学習するか左から始めるかは未確定。側脳室の予測スコアは未取得。論文の参考値はTable 2・不規則時系列1-shot LVの1.673 mmで、海馬の0.492とは別である。

## 公開templateの点検

BrainODE READMEが案内するLV_Shape_Modelingを取得・固定した。このrepositoryはLV-Netの側脳室・海馬・周辺構造を扱う処理であり、BrainODE Appendix A.1の汎用7-loss処理と同一ではない。後者を再利用する場合は違いを記録する。

配布templateのLV facesは5,117。重複4面、面積ゼロ141面があり、未処理状態では閉じたmanifoldではない。重複とゼロ面を除くと2,490頂点・4,976面の閉じた単一surfaceになり、geometry・体積は維持される。穴埋めや解剖学的平滑化は不要だった。導出templateは別ファイルとして保存し、原assetは変更していない。

公開例のSynthSegはLAS軸で、抽出codeはaffineを適用しないvoxel座標を使用する。一方、手元のtargetはRAS-worldである。導出templateは公開例から推定されるLAS→RASの軸向きを変換し、右用はmirrorした。配布template NIfTIはidentity affineで、歴史的なtemplate生成の全変換は不明である。推定と確認済み情報をtemplate_audit.jsonに分けて記録した。

## 検証

元parcellationと公開templateのSHA256を前後確認した。全mask値・native affine、VTK/OBJ再読込、voxel表面の体積、導出templateのmanifold・外向き面・非自己交差を検証した。Slicer本体でのGUI確認は未実施。

根拠：[BrainODE](https://github.com/PWonjung/BrainODE)、[著者の側脳室コード](https://github.com/PWonjung/LV_Shape_Modeling)、[BrainODE論文 Appendix A.1・Table 2](https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf)、手元atlasのmeta/info/Level5.txtとMultilevel_stripped.txt。
'''
    (PREVIEW/'README.md').write_text(docs,encoding='utf-8')
    (AUDIT/'PLAN.md').write_text(docs,encoding='utf-8')
    write_json(AUDIT/'observed_preview.json',manifest)
    checks=sorted(p for p in PREVIEW.rglob('*') if p.is_file() and p.name!='checksums.sha256')
    (PREVIEW/'checksums.sha256').write_text(''.join(digest(p)+'  '+str(p.relative_to(PREVIEW))+'\n' for p in checks))
    archive=PREVIEW.parent/'brainode_adni1_LV_observed_3d.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(p for p in PREVIEW.rglob('*') if p.is_file()):
            z.write(p,str(Path(PREVIEW.name)/p.relative_to(PREVIEW)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in sorted(p for p in PREVIEW.rglob('*') if p.is_file()):
            saved=z.read(str(Path(PREVIEW.name)/p.relative_to(PREVIEW)))
            assert hashlib.sha256(saved).hexdigest()==digest(p)
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
        template_vertices=report['assets']['left']['vertices'],template_faces=report['assets']['left']['faces'],
        observed_models=len(model_records),brainode_training_started=False,future_predictions_available=False),indent=2))


if __name__=='__main__':
    main()
