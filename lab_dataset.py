"""Lab scan eligibility and time normalization; no implicit patient exclusion.

This module records eligible observations without picking same-day repeats,
excluding single-visit subjects, choosing a split, or fitting PCA.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

from paper_cognition import observed_c


@dataclass(frozen=True)
class AgeTimeScale:
    """One common age scale fitted to eligible training observations only.

    Test/inference ages use the same affine transform, including values outside
    the training range. Patient-specific rescaling and clipping are not used.
    """

    min_age: float
    max_age: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.min_age, self.max_age)):
            raise ValueError('age bounds must be finite')
        if self.max_age <= self.min_age:
            raise ValueError('training ages must have a nonzero range')

    @classmethod
    def fit(cls, rows, *, train_subjects):
        rows = list(rows)
        subjects = set(train_subjects)
        if not subjects:
            raise ValueError('explicit training subjects are required')
        unknown = subjects - {row['pid'] for row in rows}
        if unknown:
            raise ValueError(f'training subjects have no eligible observations: {sorted(unknown)}')
        ages = [float(row['age']) for row in rows if row['pid'] in subjects]
        if not all(math.isfinite(age) for age in ages):
            raise ValueError('training ages must be finite')
        return cls(min(ages), max(ages))

    def transform(self, age):
        age = float(age)
        if not math.isfinite(age):
            raise ValueError('age must be finite')
        return (age - self.min_age) / (self.max_age - self.min_age)


def normalized_visits(rows, scale):
    return [dict(row, time=scale.transform(row['age'])) for row in rows]


def eligible_visits(rows):
    accepted, rejected = [], []
    for source in rows:
        row = dict(source)
        # A prior fixed-65/30 time must not survive a new eligibility pass.
        row.pop('time', None)
        row.pop('observed_c', None)
        reasons = []
        try:
            age = float(row['age'])
        except (KeyError, TypeError, ValueError):
            age = math.nan
        if not math.isfinite(age):
            reasons.append('missing_or_invalid_age')
        diagnosis = row.get('class', '').strip()
        if diagnosis not in ('CN', 'AD'):
            reasons.append('MCI_visit' if diagnosis == 'MCI' else 'diagnosis_not_CN_AD')
        if reasons:
            row['exclusion_reasons'] = ';'.join(reasons)
            rejected.append(row)
        else:
            row['observed_c'] = observed_c(diagnosis)
            accepted.append(row)
    accepted.sort(key=lambda r: (r['pid'], r['day'], r['uid']))
    return accepted, rejected


def write_table(path, rows, fields):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=project / 'audit/adni1_scans.csv')
    parser.add_argument('--output', type=Path, default=project / 'artifacts/adni1_allage_eligibility')
    parser.add_argument('--train-subjects', type=Path,
                        help='training patient IDs, one per line; required to fit time')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(project):
        parser.error('eligibility reports must stay under the home project')
    with args.source.open() as file:
        reader = csv.DictReader(file)
        original_fields = [f for f in reader.fieldnames if f not in ('time', 'observed_c')]
        rows = list(reader)
    accepted, rejected = eligible_visits(rows)
    scale, train_subjects = None, None
    if args.train_subjects is not None:
        train_subjects = {line.strip() for line in args.train_subjects.read_text().splitlines()
                          if line.strip()}
        scale = AgeTimeScale.fit(accepted, train_subjects=train_subjects)
        accepted = normalized_visits(accepted, scale)
    output.mkdir(parents=True, exist_ok=False)
    write_table(output / 'eligible.csv', accepted,
                original_fields + ['observed_c'] + (['time'] if scale is not None else []))
    write_table(output / 'excluded.csv', rejected, original_fields + ['exclusion_reasons'])
    patients, same_day = defaultdict(list), defaultdict(list)
    for row in accepted:
        patients[row['pid']].append(row)
        same_day[(row['pid'], row['day'])].append(row['uid'])
    summary = dict(source=str(args.source.resolve()), input_scans=len(rows),
                   eligible_scans=len(accepted), eligible_subjects=len(patients),
                   excluded_scans=len(rejected),
                   eligible_age_range=[min(float(r['age']) for r in accepted),
                                       max(float(r['age']) for r in accepted)] if accepted else None,
                   subjects_with_at_least_two_dates=sum(len({r['day'] for r in v}) >= 2
                                                        for v in patients.values()),
                   same_day_repeats=[dict(pid=p, day=d, uids=uids) for (p, d), uids in same_day.items()
                                     if len(uids) > 1],
                   rules=dict(age='all finite acquisition ages; no age-range exclusion',
                              diagnosis='CN/AD per scan; MCI visits excluded',
                              patient='retain all eligible visits regardless of MCI history',
                              time='(age - train_min_age) / (train_max_age - train_min_age); no clipping',
                              observed_c='CN=0.0, AD=1.0; pseudo continuous c comes later'),
                   unresolved=['same-day replicate policy', 'subject split', 'PCA fit population'],
                   time_status='fitted' if scale is not None else 'training subject split not supplied',
                   training_started=False)
    if scale is not None:
        parameters = dict(min_age=scale.min_age, max_age=scale.max_age,
                          span_years=scale.max_age - scale.min_age,
                          train_subjects=sorted(train_subjects),
                          train_uids=sorted(r['uid'] for r in accepted if r['pid'] in train_subjects),
                          test_and_inference='reuse these exact bounds without clipping')
        (output / 'time_scale.json').write_text(json.dumps(parameters, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in summary.items() if key != 'same_day_repeats'},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
