"""Compare source geometry against the unchanged author utilities.

Reads two engineering cases and the existing longitudinal QC artifacts.
Checks extraction, physical-coordinate round trips, ICP and topology without
changing masks, MRI images, registrations or previously fitted meshes.
"""

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import open3d as o3d
import scipy.ndimage
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_fit import PROJECT, rigid_target_to_source, sha256
from mesh_input import boundary_geometry


def author_functions():
    path = PROJECT/'references/BrainSubcorticalStructure/whole_brain_structure/util.py'
    names = {'voxel_to_bd_pcd','pcd_to_orgin','regist_mesh_pcd'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:
        raise ValueError('upstream utility functions changed')
    namespace = dict(np=np,scipy=scipy,o3d=o3d)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace,dict(path=str(path),sha256=sha256(path),
                         executed_functions=sorted(names),
                         ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[])).encode()).hexdigest())


def main():
    official,source = author_functions()
    with (PROJECT/'artifacts/multicohort_inventory/all_observations.csv').open() as file:
        rows = list(csv.DictReader(file))
    template_path = PROJECT/'dataset/Shapes/hippocampus_template_mesh.obj'
    template = trimesh.load_mesh(template_path,process=False)
    author_template_path=PROJECT/'references/BrainSubcorticalStructure/whole_brain_structure/temp_meshes/17_temp.obj'
    author_template=trimesh.load_mesh(author_template_path,process=False)
    np.testing.assert_array_equal(template.vertices,author_template.vertices)
    np.testing.assert_array_equal(template.faces,author_template.faces)
    records=[]
    for cohort,uid in [('ADNI1','34698'),('ADNI3','1426470')]:
        row, = [r for r in rows if r['cohort']==cohort and r['uid']==uid]
        path=Path(row['parcellation_path'])
        image=nib.load(path)
        mask=np.asarray(image.dataobj)==75
        g=boundary_geometry(mask,image.affine)
        author_pcd=official['voxel_to_bd_pcd'](mask)
        np.testing.assert_array_equal(np.asarray(author_pcd.points),g['boundary_voxel'])
        official['pcd_to_orgin'](author_pcd)
        np.testing.assert_allclose(np.asarray(author_pcd.points),g['target_centered_voxel_units'],atol=1e-13,rtol=0)
        world=g['boundary_header_world']
        voxel=nib.affines.apply_affine(np.linalg.inv(image.affine),world)
        np.testing.assert_allclose(voxel,g['boundary_voxel'],atol=1e-10,rtol=0)
        target,matrix,_=rigid_target_to_source(g['target_centered_header_world_units'],template.vertices)
        moving=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(g['target_centered_header_world_units']))
        fixed=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template.vertices))
        public_matrix=official['regist_mesh_pcd'](moving,fixed,normal=False,return_reg_p2p=True)
        np.testing.assert_allclose(matrix,public_matrix,atol=1e-9,rtol=0)
        inverse=np.linalg.inv(matrix)
        restored=target@inverse[:3,:3].T+inverse[:3,3]+g['centroid_header_world']
        np.testing.assert_allclose(restored,world,atol=1e-10,rtol=0)
        scale_error=abs(float(np.linalg.det(matrix[:3,:3]))-1)
        if scale_error>1e-10:
            raise ValueError('rigid ICP altered volume')
        records.append(dict(cohort=cohort,uid=uid,source_sha256=sha256(path),
            label=75,mask_voxels=int(mask.sum()),boundary_points=len(target),
            official_boundary_exact_match=True,official_centering_match=True,
            official_ICP_match=True,
            world_voxel_roundtrip_max_error=float(abs(voxel-g['boundary_voxel']).max()),
            ICP_restore_max_error=float(abs(restored-world).max()),
            ICP_volume_factor=float(np.linalg.det(matrix[:3,:3])),
            spacing=list(map(float,image.header.get_zooms()[:3])),
            spatial_unit=image.header.get_xyzt_units()[0]))
    old=PROJECT/'artifacts/mesh_qc/041_S_0898_publicedge5000'
    manifest=json.loads((old/'manifest.json').read_text())
    previous=template.vertices
    cached=[]
    for record in manifest['scans']:
        with np.load(old/f"{record['uid']}.npz") as arrays:
            np.testing.assert_array_equal(arrays['faces'],template.faces)
            np.testing.assert_array_equal(arrays['source_vertices'],previous)
            image=nib.load(record['source_path'])
            geometry=boundary_geometry(np.asarray(image.dataobj)==75,image.affine)
            matrix=arrays['icp_matrix']
            target=geometry['target_centered_header_world_units']@matrix[:3,:3].T+matrix[:3,3]
            np.testing.assert_allclose(arrays['target'],target,atol=1e-10,rtol=0)
            previous=arrays['vertices'].copy()
        cached.append(dict(uid=record['uid'],topology_exact=True,longitudinal_source_exact=True,
                           saved_target_matches_raw_mask=True))
    result=dict(date='2026-10-07 Asia/Tokyo',author_utility=source,
                template_sha256=sha256(template_path),template_vertices=len(template.vertices),
                author_left_hippocampus_template_path=str(author_template_path),
                author_left_hippocampus_template_sha256=sha256(author_template_path),
                author_left_template_vertices_and_faces_exact_match=True,
                template_faces=len(template.faces),raw_geometry_checks=records,
                existing_longitudinal_checks=cached,
                source_data_modified=False,all_cohort_quality_claim=False)
    output=PROJECT/'artifacts/mesh_qc/initialization_review'
    output.mkdir(parents=True,exist_ok=True)
    (output/'geometry_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
