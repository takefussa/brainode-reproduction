import unittest

from lab_dataset import AgeTimeScale, eligible_visits, normalized_visits
from paper_cognition import pseudo_cognitive_sample


class EligibilityTests(unittest.TestCase):
    def test_keeps_conversion_endpoints_with_mci_history(self):
        rows = [dict(pid='person', uid=str(i), day=f'200{i}-01-01', age=str(70 + i),
                     **{'class': diagnosis}) for i, diagnosis in enumerate(['CN', 'MCI', 'AD'])]
        accepted, rejected = eligible_visits(rows)
        self.assertEqual([r['uid'] for r in accepted], ['0', '2'])
        self.assertEqual([r['observed_c'] for r in accepted], [0., 1.])
        self.assertEqual(rejected[0]['exclusion_reasons'], 'MCI_visit')
        self.assertEqual(rows[0]['age'], '70')  # Input is not mutated.

    def test_all_ages_and_same_day_observations_are_preserved(self):
        rows = [dict(pid='person', uid=str(i), day='2020-01-01', age=str(age),
                     **{'class': 'CN'}) for i, age in enumerate([64., 65., 95., 96.])]
        accepted, rejected = eligible_visits(rows)
        self.assertEqual([float(r['age']) for r in accepted], [64., 65., 95., 96.])
        self.assertEqual(len(rejected), 0)
        self.assertTrue(all('time' not in row for row in accepted))

    def test_invalid_age_is_reported_and_stale_time_is_removed(self):
        rows = [dict(pid='person', uid=str(i), day='2020-01-01', age=age,
                     time='old_fixed_scale', **{'class': 'CN'})
                for i, age in enumerate(['60', '', 'nan', 'inf'])]
        accepted, rejected = eligible_visits(rows)
        self.assertEqual(len(accepted), 1)
        self.assertNotIn('time', accepted[0])
        self.assertEqual([r['exclusion_reasons'] for r in rejected],
                         ['missing_or_invalid_age'] * 3)


class AgeTimeScaleTests(unittest.TestCase):
    def test_fits_training_only_and_reuses_bounds_without_clipping(self):
        rows = [dict(pid=pid, age=age) for pid, age in
                [('train_a', 60), ('train_b', 80), ('test_a', 50), ('test_b', 90)]]
        scale = AgeTimeScale.fit(rows, train_subjects=['train_a', 'train_b'])
        self.assertEqual((scale.min_age, scale.max_age), (60., 80.))
        transformed = normalized_visits(rows, scale)
        self.assertEqual([r['time'] for r in transformed], [0., 1., -.5, 1.5])
        self.assertTrue(all('time' not in r for r in rows))
        self.assertAlmostEqual(scale.transform(71) - scale.transform(70),
                               scale.transform(66) - scale.transform(65))

    def test_missing_training_population_and_zero_range_are_errors(self):
        rows = [dict(pid='train', age=70)]
        for subjects in [[], ['unknown'], ['train']]:
            with self.subTest(subjects=subjects):
                with self.assertRaises(ValueError):
                    AgeTimeScale.fit(rows, train_subjects=subjects)
        with self.assertRaises(ValueError):
            AgeTimeScale.fit([dict(pid='train', age=float('nan'))], train_subjects=['train'])
        with self.assertRaises(ValueError):
            AgeTimeScale(60, 80).transform(float('inf'))

    def test_c_fraction_uses_patient_interval_under_training_age_scale(self):
        scale = AgeTimeScale(55., 92.)
        _, c = pseudo_cognitive_sample(
            0., 1., scale.transform(80), scale.transform(84), scale.transform(82),
            integrate=lambda shape, start, end, c: shape,
        )
        self.assertAlmostEqual(c, .5)


if __name__ == '__main__':
    unittest.main()
