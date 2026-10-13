import unittest

import numpy as np
import trimesh

from mesh_initialization import initialize_template


class TemplateInitializationTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.creation.box(extents=[4.,6.,8.])
        self.mesh.apply_translation([11.,-3.,7.])

    def test_public_unscaled_mode_preserves_exact_vertices(self):
        vertices=self.mesh.vertices.copy()
        initialized,info=initialize_template(vertices,self.mesh.faces,mode='unscaled')
        np.testing.assert_array_equal(initialized,vertices)
        self.assertEqual(info['factor'],1.)
        self.assertFalse(np.shares_memory(initialized,vertices))

    def test_size_initialization_preserves_center_indices_and_input(self):
        vertices=self.mesh.vertices.copy()
        faces=self.mesh.faces.copy()
        target_volume=self.mesh.volume*.75**3
        initialized,info=initialize_template(vertices,faces,mode='mask-volume',mask_volume=target_volume)
        fitted=trimesh.Trimesh(initialized,faces,process=False)
        self.assertAlmostEqual(fitted.volume,target_volume,places=10)
        np.testing.assert_allclose(initialized.mean(0),vertices.mean(0),atol=1e-13)
        np.testing.assert_allclose(initialized-vertices.mean(0),(vertices-vertices.mean(0))*.75)
        np.testing.assert_array_equal(vertices,self.mesh.vertices)
        np.testing.assert_array_equal(faces,self.mesh.faces)
        self.assertFalse(info['final_volume_constraint'])
        self.assertFalse(info['mask_rescaling'])

    def test_invalid_size_or_mode_is_rejected(self):
        for volume in (None,0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                initialize_template(self.mesh.vertices,self.mesh.faces,mode='mask-volume',mask_volume=volume)
        with self.assertRaises(ValueError):
            initialize_template(self.mesh.vertices,self.mesh.faces,mode='unknown')


if __name__=='__main__':
    unittest.main()
