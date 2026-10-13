import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes

from mesh_fit import FITTING_RECIPE_VERSION, WEIGHTS, mask_overlap, paper_regularizers, reconstruct_scan, reuse_prefix, rigid_target_to_source, sha256, surface_metrics, write_obj
from mesh_input import boundary_geometry


class MeshFitGeometryTests(unittest.TestCase):
    def setUp(self):
        # Regular tetrahedron: all six edge lengths are equal.
        self.vertices = torch.tensor([[1., 1., 1.], [1., -1., -1.],
                                      [-1., 1., -1.], [-1., -1., 1.]])
        self.faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        self.reference = Meshes([self.vertices], [self.faces])

    def test_rigid_translation_has_no_normal_or_edge_penalty(self):
        moved = Meshes([self.vertices + torch.tensor([2., -3., 6.])], [self.faces])
        terms = paper_regularizers(moved, self.vertices, self.reference.verts_normals_packed())
        self.assertAlmostEqual(float(terms['vertex']), 7., places=4)
        self.assertAlmostEqual(float(terms['normal']), 0., places=5)
        self.assertAlmostEqual(float(terms['edge']), 0., places=5)

    def test_normal_change_and_edge_variance_have_finite_gradients(self):
        moved = self.vertices.clone()
        moved[0, 0] += 0.5
        moved.requires_grad_()
        mesh = Meshes([moved], [self.faces])
        terms = paper_regularizers(mesh, self.vertices, self.reference.verts_normals_packed())
        self.assertGreater(float(terms['normal']), 0.)
        lengths = np.linalg.norm(moved.detach().numpy()[[0, 0, 0, 1, 1, 2]] -
                                 moved.detach().numpy()[[1, 2, 3, 2, 3, 3]], axis=1)
        self.assertAlmostEqual(float(terms['edge']), float(np.var(lengths)), places=5)
        sum(terms.values()).backward()
        self.assertTrue(torch.isfinite(moved.grad).all())
        self.assertGreater(float(moved.grad.norm()), 0.)

    def test_public_edge_variant_matches_the_author_source(self):
        path = Path(__file__).resolve().parents[1] / 'references/BrainSubcorticalStructure/models/loss_3d.py'
        spec = importlib.util.spec_from_file_location('author_edge_loss', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        moved = self.vertices.clone()
        moved[0, 0] += 0.5
        mesh = Meshes([moved], [self.faces])
        ours = paper_regularizers(mesh, self.vertices, self.reference.verts_normals_packed(), 'public')['edge']
        torch.testing.assert_close(ours, module.mesh_edge_var_loss(mesh))

    def test_icp_preserves_size_and_recovers_small_rigid_motion(self):
        rng = np.random.default_rng(17)
        source = rng.normal(size=(200, 3)) * [2, 4, 1]
        angle = 0.05
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        target = source @ rotation.T + [0.1, -0.1, 0.2]
        aligned, matrix, info = rigid_target_to_source(target, source)
        np.testing.assert_allclose(aligned, source, atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.det(matrix[:3, :3])), 1., places=6)
        self.assertEqual(info['fitness'], 1.)

    def test_overlap_restores_affine_centroid_and_icp(self):
        mask = np.zeros((7, 7, 7), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        affine = np.diag([2., 2., 2., 1.])
        affine[:3, 3] = [13., -6., 8.]
        geometry = boundary_geometry(mask, affine)
        box = trimesh.creation.box(extents=[2., 2., 2.])
        voxel = np.asarray(box.vertices) + [2.5, 2.5, 2.5]
        centered = voxel @ affine[:3, :3].T + affine[:3, 3] - geometry['centroid_header_world']
        matrix = np.array([[0., -1., 0., 2.], [1., 0., 0., -3.],
                           [0., 0., 1., 1.], [0., 0., 0., 1.]])
        canonical = centered @ matrix[:3, :3].T + matrix[:3, 3]
        overlap = mask_overlap(canonical, np.asarray(box.faces), geometry, matrix, mask)
        self.assertEqual(overlap['dice'], 1.)
        self.assertEqual(overlap['reconstructed_voxels'], 8)

    def test_saved_geometry_uses_final_optimizer_parameters(self):
        class Translation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.tensor([1., 2., 3.]))

            def forward(self, points):
                return self.offset[None, :, None].expand(1, 3, points.shape[1])

        model = Translation()

        def distance_loss(mesh, target, reference_vertices, reference_normals, edge_mode):
            loss = (mesh.verts_packed()-reference_vertices).square().mean()
            terms = {key:loss.new_zeros(()) for key in WEIGHTS}
            terms['vertex'] = loss
            return loss, terms

        with tempfile.TemporaryDirectory() as temporary, patch('mesh_fit.official_pointnet', return_value=model):
            vertices, _ = reconstruct_scan(
                self.vertices.numpy(), self.faces.numpy(), self.vertices.numpy(), 1,
                'cpu', 42, Path(temporary)/'loss.csv', 'fixed-source', 'public',
                loss_function=distance_loss)
        expected = self.vertices.numpy()+model.offset.detach().numpy()
        np.testing.assert_allclose(vertices, expected, rtol=0, atol=1e-7)
        self.assertFalse(np.allclose(vertices, self.vertices.numpy()+[1., 2., 3.]))

    def test_resume_rejects_recipe_change_and_restores_exact_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, output = root / 'cache', root / 'output'
            cache.mkdir(); output.mkdir()
            mask_path = root / 'mask_source'
            mask_path.write_bytes(b'synthetic source for hash validation')
            original, faces = self.vertices.numpy(), self.faces.numpy()
            candidate = original + .1
            settings = dict(fitting_recipe_version=FITTING_RECIPE_VERSION,
                            initialization='unscaled',initialization_info={'factor':1.},
                            subject='person', iterations=5000, seed=42,
                            update_mode='fixed-source', edge_mode='public', weights={'edge': 1500.},
                            template_sha256='template', face_array_sha256='faces',
                            spatial_unit_assumption='1 numeric unit = 1 mm',
                            code_provenance={'mesh_fit.py':'a'*64}, environment={'torch':'synthetic'},
                            observation_ordering='synthetic order', scan_table_sha256='table')
            record = dict(uid='scan1', day='2001-01-01', age=70., diagnosis='CN',
                          source_sha256=sha256(mask_path))
            np.savez(cache / 'scan1.npz', source_vertices=original, vertices=candidate, faces=faces)
            write_obj(cache / 'scan1.obj', candidate, faces)
            (cache / 'scan1_loss.csv').write_text('iteration,total\n5000,0\n')
            record['artifact_sha256']={name:sha256(cache/name) for name in ('scan1.npz','scan1.obj','scan1_loss.csv')}
            (cache / 'manifest.json').write_text(json.dumps(dict(settings, scans=[record])))
            rows = [dict(pid='person', uid='scan1', day='2001-01-01', age='70',
                         parcellation_path=str(mask_path), **{'class': 'CN'})]
            changed = dict(settings, edge_mode='paper-variance', scans=[])
            with self.assertRaisesRegex(ValueError, 'edge_mode'):
                reuse_prefix(cache, output, changed, rows, original, faces)
            self.assertEqual(list(output.iterdir()), [])
            different_initialization = dict(settings, initialization='mask-volume', scans=[])
            with self.assertRaisesRegex(ValueError, 'initialization'):
                reuse_prefix(cache, output, different_initialization, rows, original, faces)
            legacy = dict(settings, fitting_recipe_version=1, scans=[])
            with self.assertRaisesRegex(ValueError, 'fitting_recipe_version'):
                reuse_prefix(cache, output, legacy, rows, original, faces)
            accepted = dict(settings, scans=[])
            restored, start = reuse_prefix(cache, output, accepted, rows, original, faces)
            np.testing.assert_array_equal(restored, candidate)
            self.assertEqual(start, 1)
            self.assertTrue((output / 'scan1.npz').exists())
            self.assertEqual(accepted['reused_prefix']['scans'], 1)
            different_code = dict(settings,code_provenance={'mesh_fit.py':'b'*64},scans=[])
            with self.assertRaisesRegex(ValueError,'code_provenance'):
                reuse_prefix(cache,output,different_code,rows,original,faces)
            in_place = dict(settings,scans=[])
            restored,start = reuse_prefix(cache,cache,in_place,rows,original,faces)
            np.testing.assert_array_equal(restored,candidate)
            self.assertEqual(start,1)
            (cache/'scan1.obj').write_text('corrupted artifact')
            with self.assertRaisesRegex(ValueError,'artifact hash'):
                reuse_prefix(cache,output,dict(settings,scans=[]),rows,original,faces)

    def test_watertightness_does_not_hide_intersecting_surfaces(self):
        box = trimesh.creation.box(extents=[2., 2., 2.])
        clean = surface_metrics(box.vertices, box.faces, box.vertices)
        self.assertTrue(clean['watertight'])
        self.assertEqual(clean['self_intersecting_triangle_pairs'], 0)
        shifted = box.copy()
        shifted.apply_translation([1., .3, .2])
        crossed = trimesh.util.concatenate([box, shifted])
        result = surface_metrics(crossed.vertices, crossed.faces, crossed.vertices)
        self.assertTrue(result['watertight'])
        self.assertGreater(result['self_intersecting_triangle_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
