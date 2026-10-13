"""Read-only ADNI1 inventory. Writes reports only inside this project's audit/.

All NIfTI headers are inspected. Voxel labels are inspected for 12 deterministic
samples, explicitly recorded in the output; no mesh, PCA or model is produced.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date
import csv
import hashlib
import json
import math

import nibabel as nib
import numpy as np

ROOT = Path('/home/take/brainode')
SOURCE = Path('/data2/radiology_datas/clean3')
OUT = ROOT / 'audit'
assert ROOT.resolve() == ROOT and OUT.resolve().is_relative_to(ROOT)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def finite(x):
    try:
        return math.isfinite(float(x))
    except (ValueError, TypeError):
        return False


def stats(xs):
    x = np.array(list(xs), dtype=float)
    return {'n': len(x), 'min': float(x.min()), 'p25': float(np.percentile(x, 25)),
            'median': float(np.median(x)), 'p75': float(np.percentile(x, 75)),
            'max': float(x.max()), 'mean': float(x.mean()), 'std_population': float(x.std())} if len(x) else {'n': 0}


def distribution(values):
    return dict(sorted(Counter(values).items()))


meta_path = SOURCE / 'meta/csv/ADNI1/meta.csv'
rows = list(csv.DictReader(meta_path.open()))
json_rows = json.loads((SOURCE / 'meta/json/ADNI1.json').read_text())
by_uid = defaultdict(list)
by_pid = defaultdict(list)
for row in rows:
    by_uid[row['uid']].append(row)
    by_pid[row['pid']].append(row)

sets = {}
paths = {}
for kind, rel in [('trsaa', 'full/trsaa'), ('rigid', 'full/rigid'), ('parcellation', 'parcellation')]:
    paths[kind] = {p.stem: p for p in sorted((SOURCE / 'ADNI1' / rel).glob('*.nii'))}
    sets[kind] = set(paths[kind])

summary = {'metadata_path': str(meta_path), 'metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
           'metadata_rows': len(rows), 'metadata_unique_uid': len(by_uid), 'subjects': len(by_pid),
           'duplicate_uid': {k: len(v) for k, v in by_uid.items() if len(v) > 1},
           'file_counts': {k: len(v) for k, v in paths.items()},
           'uid_set_equals_metadata': {k: v == set(by_uid) for k, v in sets.items()},
           'csv_json_equal': rows == json_rows,
           'metadata_path_full_mni_matches_trsaa': all(row['path_full_mni'] == str(paths['trsaa'].get(row['uid'])) for row in rows),
           'columns': list(rows[0]),
           'missing': {k: sum(row[k].strip().lower() in ('', 'nan', 'none', 'null') for row in rows) for k in rows[0]},
           'class_counts': distribution(row['class'] for row in rows),
           'class_base_counts': distribution(row['class_base'] for row in rows),
           'age': stats(float(row['age']) for row in rows if finite(row['age'])),
           'day_range': [min(row['day'] for row in rows), max(row['day'] for row in rows)],
           'numeric_age_outside_65_95': sum(not 65 <= float(row['age']) <= 95 for row in rows if finite(row['age']))}
subjects = []
intervals = []
unique_intervals = []
same_day_groups = []
age_date_difference = []
for pid, group in sorted(by_pid.items()):
    group.sort(key=lambda row: (row['day'], int(row['uid'])))
    dates = [date.fromisoformat(row['day']) for row in group]
    days = sorted(set(dates))
    intervals.extend((b - a).days for a, b in zip(dates, dates[1:]))
    unique_intervals.extend((b - a).days for a, b in zip(days, days[1:]))
    for a, b in zip(group, group[1:]):
        age_date_difference.append((float(b['age'])-float(a['age'])) - (date.fromisoformat(b['day'])-date.fromisoformat(a['day'])).days/365.25)
    grouped_days = defaultdict(list)
    for row in group:
        grouped_days[row['day']].append(row['uid'])
    for day, uids in grouped_days.items():
        if len(uids) > 1:
            same_day_groups.append({'pid': pid, 'day': day, 'uids': uids})
    subjects.append({'pid': pid, 'files': len(group), 'distinct_dates': len(days),
                     'first_day': min(row['day'] for row in group), 'last_day': max(row['day'] for row in group),
                     'classes': ','.join(sorted(set(row['class'] for row in group))),
                     'first_age': group[0]['age'], 'last_age': group[-1]['age'],
                     'all_age_65_95': all(65 <= float(row['age']) <= 95 for row in group)})

summary.update({'file_count_distribution': distribution(s['files'] for s in subjects),
                'distinct_date_distribution': distribution(s['distinct_dates'] for s in subjects),
                'at_least_n_files': {str(n): sum(s['files'] >= n for s in subjects) for n in range(2, 6)},
                'at_least_n_distinct_dates': {str(n): sum(s['distinct_dates'] >= n for s in subjects) for n in range(2, 6)},
                'adjacent_file_intervals_days': stats(intervals), 'positive_unique_date_intervals_days': stats(unique_intervals),
                'most_common_positive_intervals_days': Counter(unique_intervals).most_common(15),
                'zero_interval_pairs': intervals.count(0), 'same_day_groups': same_day_groups,
                'age_increment_minus_date_increment_years': stats(age_date_difference),
                'diagnosis_sequence_sets': distribution(s['classes'] for s in subjects),
                'stable_CN_or_AD_at_least_two_dates': sum(s['classes'] in ('CN', 'AD') and s['distinct_dates'] >= 2 for s in subjects),
                'stable_CN_or_AD_at_least_two_dates_all_age_65_95': sum(s['classes'] in ('CN', 'AD') and s['distinct_dates'] >= 2 and s['all_age_65_95'] for s in subjects)})
dump('adni1_metadata_summary.json', summary)
for name, table in [('adni1_subjects.csv', subjects), ('adni1_scans.csv', [{**row, 'parcellation_path': str(paths['parcellation'].get(row['uid']))} for row in rows])]:
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
print('Metadata:', len(rows), 'scans;', len(subjects), 'subjects', flush=True)

geometries = {}
loaded = {}
for kind, items in paths.items():
    counts = Counter()
    geometry_by_uid = {}
    for uid, path in items.items():
        image = nib.load(path)
        h = image.header
        geometry = {'shape': list(image.shape), 'spacing': list(map(float, h.get_zooms())),
                    'affine': image.affine.tolist(), 'orientation': list(nib.aff2axcodes(image.affine)),
                    'dtype': str(h.get_data_dtype()), 'qform_code': int(h['qform_code']),
                    'sform_code': int(h['sform_code']), 'units': list(h.get_xyzt_units()),
                    'description': bytes(h['descrip']).rstrip(b'\x00').decode(errors='replace')}
        signature = json.dumps(geometry, sort_keys=True)
        counts[signature] += 1
        geometry_by_uid[uid] = geometry
    loaded[kind] = geometry_by_uid
    geometries[kind] = {'n': len(items), 'geometry_signatures': [{'count': count, **json.loads(key)} for key, count in counts.most_common()]}
    print('Headers inspected:', kind, len(items), 'signatures', len(counts), flush=True)

comparisons = {}
for other in ('trsaa', 'rigid'):
    common = sorted(sets['parcellation'] & sets[other])
    same_shape = [u for u in common if loaded['parcellation'][u]['shape'] == loaded[other][u]['shape']]
    same_affine = [u for u in common if np.allclose(loaded['parcellation'][u]['affine'], loaded[other][u]['affine'], atol=1e-5, rtol=0)]
    comparisons[other] = {'pairs': len(common), 'same_shape': len(same_shape), 'same_affine': len(same_affine),
                          'same_shape_and_affine': len(set(same_shape) & set(same_affine)),
                          'example_uid': common[0], 'example_mri': loaded[other][common[0]],
                          'example_parcellation': loaded['parcellation'][common[0]]}
dump('adni1_geometry.json', {'datasets': geometries, 'parcellation_vs_mri': comparisons})

uids = sorted(sets['parcellation'], key=int)
sample_uids = [uids[i] for i in np.linspace(0, len(uids)-1, 12, dtype=int)]
volumes = {row['uid']: row for row in csv.DictReader((SOURCE / 'meta/csv/ADNI1/volume_full.csv').open())}
label_samples = []
label_volume_validations = []
lookup = {int(s.split()[0]): s.split()[1] for s in (SOURCE / 'meta/level/Level5.txt').read_text().splitlines()}
for uid in sample_uids:
    image = nib.load(paths['parcellation'][uid])
    data = np.asanyarray(image.dataobj)
    labels, counts = np.unique(data, return_counts=True)
    points = np.argwhere(data == 75)
    world_mean = nib.affines.apply_affine(image.affine, points.mean(0))
    count_by_label = dict(zip(map(str, labels), map(int, counts)))
    label_samples.append({'uid': uid, 'unique_labels': labels.tolist(), 'counts': count_by_label,
                          'left_hippocampus_voxels': len(points), 'assumed_mm3_if_spacing_mm': float(len(points) * np.prod(image.header.get_zooms())),
                          'spatial_units_header': image.header.get_xyzt_units()[0],
                          'Hippo_L_volume_full_csv': volumes[uid]['Hippo_L'],
                          'left_hippocampus_voxel_bbox': [points.min(0).tolist(), points.max(0).tolist()],
                          'left_hippocampus_world_centroid': world_mean.tolist()})
    mismatches = [{'code': code, 'roi': roi, 'voxels': count_by_label.get(str(code), 0), 'csv': float(volumes[uid][roi])}
                  for code, roi in lookup.items() if count_by_label.get(str(code), 0) != float(volumes[uid][roi])]
    label_volume_validations.append({'uid': uid, 'regional_volume_count_mismatches': mismatches})
    print('Labels inspected:', uid, len(labels), 'labels; Hippo_L', len(points), 'voxels', flush=True)
dump('adni1_label_samples.json', {'scope': '12 evenly spaced UIDs sorted numerically, not all scans', 'samples': label_samples,
                                'union': sorted(set(x for sample in label_samples for x in sample['unique_labels']))})
dump('adni1_label_volume_validation.json', label_volume_validations)
print('Read-only audit complete.', flush=True)
