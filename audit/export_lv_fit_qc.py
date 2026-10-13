"""Add the completed one-scan LV PointNet QC to the observed 3D package."""

import argparse
import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import trimesh

from export_review_bundle import digest, voxel_boundary, write_mesh, write_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mesh-directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.mesh_directory.resolve()
    package = PROJECT/'artifacts/review_bundles/adni1_LV_RID352_observed'
    original = json.loads((directory/'manifest.json').read_text())
    if original['subject'] != 'ADNI:RID:352' or original['region'] != 'lateral_ventricle_left':
        raise ValueError('unexpected QC subject or side')
    record = next(r for r in original['scans'] if r['uid']=='34537')
    npz = directory/'34537.npz'
    if digest(npz) != record['artifact_sha256'][npz.name]:
        raise ValueError('QC mesh changed')
    with np.load(npz, allow_pickle=False) as data:
        canonical = data['vertices'].astype(np.float64)
        faces = data['faces'].copy()
        inverse = np.linalg.inv(data['icp_matrix'])
        inverse[:3,3] += data['centroid_header_world']
        native = nib.affines.apply_affine(inverse, canonical)
        np.testing.assert_allclose(nib.affines.apply_affine(np.linalg.inv(inverse),native),
                                   canonical, atol=1e-11, rtol=0)
    np.testing.assert_allclose(trimesh.Trimesh(native,faces,process=False).volume,
                               trimesh.Trimesh(canonical,faces,process=False).volume,
                               atol=1e-8, rtol=1e-10)
    vtk,obj = write_mesh(package/'data/baseline_left_LV_fitted_mesh',native,faces)
    image=nib.load(package/'data/baseline_left_LV_segmentation.nii.gz')
    boundary, boundary_faces=voxel_boundary(np.asarray(image.dataobj).astype(bool),image.affine)
    fig=go.Figure()
    for title,v,f,color,opacity in [
        ('Observed left LV segmentation voxel boundary',boundary,boundary_faces,'#3377db',0.25),
        ('PointNet fitted left LV mesh',native,faces,'#ed842e',0.80)]:
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            color=color,opacity=opacity,name=title,showlegend=True))
    fig.update_layout(title='初回左側脳室：segmentationとPointNet mesh',height=760,
        scene=dict(aspectmode='data',xaxis_title='R（仮定mm）',yaxis_title='A（仮定mm）',zaxis_title='S（仮定mm）'))
    score=f"Dice={record['overlap']['dice']:.4f}、体積差={record['volume_bias_percent']:+.3f}%、自己交差triangle-pairs={record['after']['self_intersecting_triangle_pairs']}。"
    (package/'fit_comparison.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8">'
        '<title>側脳室 PointNet fit</title><body style="font-family:sans-serif"><h1>左側脳室のmesh fit</h1>'
        '<p>RID352・2006-04-07。青＝実測segmentationのvoxel境界、橙＝固定topologyのPointNet fit。'
        '全てnative parcellation RAS。</p><p>'+score+' このmeshは初回の観測shapeで、将来予測ではない。</p>'+
        fig.to_html(full_html=False,include_plotlyjs=True)+'</body></html>',encoding='utf-8')
    result=dict(uid='34537',subject=original['subject'],region=original['region'],
        stage='one-scan engineering QC, not cohort BrainODE training',
        vertex_count=len(native),face_count=len(faces),iterations=original['iterations'],
        dice=record['overlap']['dice'],volume_bias_percent=record['volume_bias_percent'],
        self_intersection_pairs=record['after']['self_intersecting_triangle_pairs'],
        qc_flags=record['qc_flags'],mask_components=record['mask_components_26'],
        original_mesh_npz=str(npz),original_mesh_sha256=digest(npz),
        vtk=str(vtk.relative_to(package)),obj=str(obj.relative_to(package)),
        coordinate_system='native baseline parcellation RAS',
        original_source_sha256=record['source_sha256'],
        coordinate_roundtrip_verified=True,volume_preserved=True,
        future_prediction=False,voxel_resampling=False,mesh_repair=False)
    write_json(package/'pointnet_fit_qc.json',result)
    manifest=json.loads((package/'preview_manifest.json').read_text())
    manifest['fitted_correspondence_mesh_available']=True
    manifest['fitted_mesh_scope']='one baseline left LV scan only; other surfaces remain voxel boundaries'
    manifest['pointnet_fit_qc']=result
    write_json(package/'preview_manifest.json',manifest)
    text=(package/'README.md').read_text()
    text=text.replace(
        'このパッケージのsurfaceは、foreground voxelの露出面を三角形で表示するルールベースの表面である。対応付きPointNet meshでも、BrainODEによる予測でもない。',
        '`*_observed_voxel_surface`は、foreground voxelの露出面を三角形で表示するルールベースの補助表面である。別ファイルの`baseline_left_LV_fitted_mesh`は初回左側のPointNet対応付きmeshである。いずれもBrainODEによる将来予測ではない。')
    marker='## 初回左側脳室のPointNet fit追加'
    if marker in text:
        text=text.split(marker)[0].rstrip()+'\n'
    text+='\n'+marker+'\n\n'+'1撮像（初回UID34537、左側）で5,000 updatesのfitを完了した。'+score+\
        '小成分は全てtargetに保持し、自己交差は修復していない。1撮像のQCであり、全cohortの品質や将来予測の精度を検証したものではない。\n\n'+\
        '`data/baseline_left_LV_fitted_mesh.vtk`をSlicerのModelとして開く。`fit_comparison.html`は実測maskとの回転可能な比較。'+\
        '予測モデル用の固定頂点対応があるmeshを作れることを確認したが、側脳室用PCA・CNN・BrainODEの本学習と将来スコアは未取得。\n'
    (package/'README.md').write_text(text,encoding='utf-8')
    files=sorted(p for p in package.rglob('*') if p.is_file() and p.name!='checksums.sha256')
    (package/'checksums.sha256').write_text(''.join(digest(p)+'  '+str(p.relative_to(package))+'\n' for p in files))
    archive=package.parent/'brainode_adni1_LV_observed_3d.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(p for p in package.rglob('*') if p.is_file()):
            z.write(p,str(Path(package.name)/p.relative_to(package)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in sorted(p for p in package.rglob('*') if p.is_file()):
            assert z.read(str(Path(package.name)/p.relative_to(package)))==p.read_bytes()
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
