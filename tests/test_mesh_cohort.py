import unittest
from mesh_cohort import ordered_scans, review_flags, subject_folder


class MeshCohortTests(unittest.TestCase):
    def test_same_day_repeat_is_kept_with_numeric_order_and_cross_phase(self):
        rows = [dict(pid='ADNI:RID:4',uid=u,day=d,age=a,**{'class':'CN'},cohort=c)
                for u,d,a,c in [('10','2012-01-01','70','ADNI2'),
                                ('3','2013-01-01','71','ADNI3'),
                                ('2','2012-01-01','70','ADNI2')]]
        result = ordered_scans(rows)
        self.assertEqual([r['uid'] for r in result], ['2','10','3'])
        self.assertEqual(len(result), len(rows))
        self.assertEqual(rows[0]['uid'], '10')

    def test_conflicting_repeat_or_identifier_collision_is_rejected(self):
        a = dict(pid='person',uid='1',day='2012-01-01',age='70',**{'class':'CN'})
        b = dict(a,uid='2',age='71')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            ordered_scans([a,b])
        with self.assertRaisesRegex(ValueError, 'collision'):
            ordered_scans([a,a])
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            ordered_scans([dict(a,uid='../scan')])

    def test_review_flags_preserve_acceptable_volume_with_bad_topology(self):
        record = dict(after=dict(watertight=True,winding_consistent=True,volume_mm3=3000.,
                                 self_intersecting_triangle_pairs=2,boundary_to_surface_mean_mm=.4),
                      overlap={'dice':.95},volume_bias_percent=.2,
                      mask_components_26=1,mask_touches_image_edge=False)
        self.assertEqual(review_flags(record), ['self_intersection'])
        self.assertEqual(record['after']['volume_mm3'], 3000.)

    def test_safe_subject_folder_does_not_collapse_distinct_keys(self):
        self.assertNotEqual(subject_folder('a:b'),subject_folder('a_b'))
        self.assertNotIn('/',subject_folder('../a:b'))


if __name__ == '__main__':
    unittest.main()
