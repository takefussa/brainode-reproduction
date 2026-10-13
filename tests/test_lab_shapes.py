import unittest

import numpy as np

from lab_shapes import fit_train_pca, patient_indices


class LabShapesTests(unittest.TestCase):
    def test_heldout_geometry_cannot_change_pca_basis(self):
        rng=np.random.default_rng(8)
        vertices=rng.normal(size=(9,4,3))
        pids=np.array(['a','a','b','b','c','c','d','e','e'])
        splits=np.array(['train']*7+['test']*2)
        first,_,_=fit_train_pca(vertices,pids,splits,3)
        altered=vertices.copy();altered[7:]+=10000
        second,_,_=fit_train_pca(altered,pids,splits,3)
        np.testing.assert_array_equal(first.mean_,second.mean_)
        np.testing.assert_array_equal(first.components_,second.components_)

    def test_patient_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'multiple splits'):
            patient_indices(['a','b','a'],['train','train','test'])

    def test_replicates_and_single_observations_remain_in_dataset(self):
        groups=patient_indices(['a','a','b','a'],['train']*3+['train'])
        np.testing.assert_array_equal(groups['a'],[0,1,3])
        np.testing.assert_array_equal(groups['b'],[2])


if __name__=='__main__':
    unittest.main()
