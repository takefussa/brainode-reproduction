"""Prepare an ADNI1-only stage while preserving the existing RID split."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lab_dataset import AgeTimeScale, normalized_visits, write_table
from mesh_cohort import ordered_scans


def main():
    source = PROJECT/'artifacts/adni_all_baseline'
    output = PROJECT/'artifacts/adni1_stage_baseline'
    tables, fields = {}, {}
    for name in ('eligible', 'excluded'):
        with (source/f'{name}.csv').open() as file:
            reader = csv.DictReader(file)
            fields[name] = reader.fieldnames
            tables[name] = [row for row in reader if row['cohort'] == 'ADNI1']
    rows = tables['eligible']
    patients = defaultdict(list)
    for row in rows:
        patients[row['pid']].append(row)
        if not Path(row['parcellation_path']).is_file() or not Path(row['trsaa_path']).is_file():
            raise ValueError('required source file missing')
    for group in patients.values():
        ordered_scans(group)
        if len({row['split'] for row in group}) != 1:
            raise ValueError('patient occurs in both train and test')
    splits = {split: {r['pid'] for r in rows if r['split'] == split} for split in ('train', 'test')}
    if splits['train'] & splits['test'] or splits['train'] | splits['test'] != set(patients):
        raise ValueError('invalid inherited patient split')
    if len({r['scan_key'] for r in rows}) != len(rows):
        raise ValueError('duplicate observation key')
    scale = AgeTimeScale.fit(rows, train_subjects=splits['train'])
    normalized = normalized_visits(rows, scale)
    for original, updated in zip(rows, normalized):
        if {k: v for k, v in updated.items() if k != 'time'} != {k: v for k, v in original.items() if k != 'time'}:
            raise ValueError('non-time metadata changed')
    conversions = {split: [] for split in splits}
    for pid, group in patients.items():
        ordered = ordered_scans(group)
        if any(a['class'] == 'CN' and b['class'] == 'AD' and float(a['age']) < float(b['age'])
               and a['day'] < b['day'] for index, a in enumerate(ordered) for b in ordered[index+1:]):
            conversions[group[0]['split']].append(pid)
    parameters = dict(min_age=scale.min_age, max_age=scale.max_age,
                      span_years=scale.max_age-scale.min_age,
                      fit_scope='ADNI1 eligible training observations only',
                      train_subjects=sorted(splits['train']),
                      train_scan_keys=sorted(r['scan_key'] for r in rows if r['split'] == 'train'),
                      test_and_inference='reuse without refitting or clipping')
    summary = dict(cohort='ADNI1', source_baseline=str(source),
        source_hashes={f'{name}.csv': hashlib.sha256((source/f'{name}.csv').read_bytes()).hexdigest()
                       for name in tables},
        eligible_subjects=len(patients), eligible_scans=len(rows), excluded_scans=len(tables['excluded']),
        total_source_scans=len(rows)+len(tables['excluded']),
        class_counts=dict(Counter(r['class'] for r in rows)),
        split_subjects={s: len(pids) for s, pids in splits.items()},
        split_scans=dict(Counter(r['split'] for r in rows)),
        split_policy='inherit merged-RID ADNI1/2/3 seed42 split; do not redraw an ADNI1-only split',
        age_range=[min(float(r['age']) for r in rows), max(float(r['age']) for r in rows)],
        train_time_scale={k: parameters[k] for k in ('min_age','max_age','span_years')},
        same_day_repeat_groups=sum(n > 1 for n in Counter((r['pid'], r['day']) for r in rows).values()),
        subjects_with_at_least_two_dates=sum(len({r['day'] for r in group}) >= 2 for group in patients.values()),
        single_scan_subjects=sum(len(group) == 1 for group in patients.values()),
        CN_to_AD_subject_keys=conversions,
        eligibility='all finite-age CN/AD visits; MCI visits excluded; no age cutoffs or additional selection',
        repeats='all preserved at unchanged observation ages and dates',
        time_note='ADNI1 train-only coefficients replace full-ADNI coefficients; mesh fitting does not use time',
        mesh_quality_adoption=False, PCA_fitted=False, ODE_CNN_training_started=False,
        original_sources_modified=False)
    output.mkdir(parents=True, exist_ok=False)
    write_table(output/'eligible.csv', normalized, fields['eligible'])
    write_table(output/'excluded.csv', tables['excluded'], fields['excluded'])
    for split, pids in splits.items():
        (output/f'{split}_subjects.txt').write_text('\n'.join(sorted(pids))+'\n')
    for name, value in [('time_scale', parameters), ('summary', summary)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
