"""Read-only review of six locally available BrainODE cohorts.

Retains every metadata row and reports literal CN/AD + finite-age counts.
These counts are not a finalized cohort selection or diagnosis mapping.
"""

from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path('/data2/radiology_datas/clean3')
OUTPUT = PROJECT / 'artifacts/multicohort_inventory'
COHORTS = ['ADNI1', 'ADNI2', 'ADNI3', 'AIBL', 'OASIS2', 'OASIS3']


def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def patient_key(cohort, pid):
    if cohort.startswith('ADNI'):
        match = re.fullmatch(r'\d{3}_S_(\d+)', pid)
        if match is None:
            raise ValueError(f'unrecognized ADNI PTID: {pid}')
        return f'ADNI:RID:{int(match.group(1))}'
    return f'{cohort}:{pid}'


def time_field(cohort):
    return ('date_image' if cohort == 'AIBL' else
            'mr_delay' if cohort == 'OASIS2' else 'day')


def parse_date(value, date_format):
    try:
        return datetime.strptime(value, date_format)
    except (ValueError, TypeError):
        return None


def ordered_time(cohort, row):
    value = row[time_field(cohort)]
    if cohort.startswith('ADNI') or cohort == 'AIBL':
        # AIBL slash dates contain days >12 but no first-field value >12.
        date = parse_date(value, '%m/%d/%Y' if cohort == 'AIBL' else '%Y-%m-%d')
        return date.toordinal() if date is not None else math.nan
    return float(value) if finite(value) else math.nan


def oasis2_session(uid):
    # One scan uses suffix _3906-3 instead of _mpr-N; retain the MR session.
    match = re.match(r'^(OAS2_\d+_MR\d+)_', uid)
    if match is None:
        raise ValueError(f'unrecognized OASIS2 scan ID: {uid}')
    return match.group(1)


def sample_mask(path):
    image = nib.load(path)
    volume = np.asanyarray(image.dataobj)
    result = dict(path=str(path), shape=list(image.shape), dtype=str(image.get_data_dtype()),
                  spacing=list(map(float, image.header.get_zooms()[:3])),
                  units=image.header.get_xyzt_units()[0], orientation=''.join(nib.aff2axcodes(image.affine)),
                  max_label=int(volume.max()), labels={})
    for label in (75, 76):
        points = np.argwhere(volume == label)
        record = dict(voxels=len(points))
        if len(points):
            minimum, maximum = points.min(0), points.max(0)
            crop = volume[tuple(slice(a, b+1) for a,b in zip(minimum, maximum))] == label
            _, components = ndimage.label(crop, structure=np.ones((3,3,3), dtype=bool))
            record.update(voxel_bbox=[minimum.tolist(), maximum.tolist()],
                          header_world_centroid=nib.affines.apply_affine(image.affine, points.mean(0)).tolist(),
                          components_26=int(components))
        result['labels'][str(label)] = record
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, records, adni_rows, samples = {}, [], [], []
    all_by_patient = defaultdict(list)
    for cohort in COHORTS:
        meta = SOURCE / 'meta/csv' / cohort / 'meta.csv'
        rows = list(csv.DictReader(meta.open()))
        masks = {path.stem: path for path in (SOURCE / cohort / 'parcellation').glob('*.nii')}
        mris = {path.stem: path for path in (SOURCE / cohort / 'full/trsaa').glob('*.nii')}
        by_uid = Counter(row['uid'] for row in rows)
        by_patient, candidates = defaultdict(list), []
        for row in rows:
            key = patient_key(cohort, row['pid'])
            reasons = []
            if row['class'] not in ('CN','AD'):
                reasons.append('MCI' if row['class'] == 'MCI' else 'diagnosis_requires_mapping_or_is_missing')
            if not finite(row['age']):
                reasons.append('age_missing_or_invalid')
            if row['uid'] not in masks:
                reasons.append('parcellation_missing')
            if row['uid'] not in mris:
                reasons.append('TRSAA_missing')
            if not math.isfinite(ordered_time(cohort, row)):
                reasons.append('scan_time_missing_or_invalid')
            record = dict(cohort=cohort,pid=row['pid'],patient_key=key,uid=row['uid'],
                          scan_key=f"{cohort}:{row['uid']}",age=row['age'],diagnosis=row['class'],
                          time_field=time_field(cohort),time_value=row[time_field(cohort)],
                          parcellation_path=str(masks[row['uid']]) if row['uid'] in masks else '',
                          trsaa_path=str(mris[row['uid']]) if row['uid'] in mris else '',
                          literal_rules_match=row['class'] in ('CN', 'AD') and finite(row['age']),
                          literal_rules_files_and_time_match=not reasons,review_reasons=';'.join(reasons),
                          raw_metadata=json.dumps(row,ensure_ascii=False))
            records.append(record)
            by_patient[key].append(row)
            all_by_patient[key].append(record)
            if not reasons:
                candidates.append(row)
            if cohort.startswith('ADNI'):
                adni_rows.append(dict(row,cohort=cohort,patient_key=key))
        candidate_by_patient = defaultdict(list)
        for row in candidates:
            candidate_by_patient[patient_key(cohort,row['pid'])].append(row)
        summary = dict(metadata_path=str(meta), metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),
                       metadata_rows=len(rows),subjects=len(by_patient),class_counts=dict(Counter(r['class'] for r in rows)),
                       parcellation_files=len(masks),trsaa_files=len(mris),duplicate_metadata_uid={u:n for u,n in by_uid.items() if n>1},
                       missing_parcellations=[r['uid'] for r in rows if r['uid'] not in masks],
                       missing_TRSAA=[r['uid'] for r in rows if r['uid'] not in mris],
                       literal_CN_AD_rows=sum(r['class'] in ('CN','AD') for r in rows),
                       finite_age_literal_CN_AD_rows=sum(r['class'] in ('CN','AD') and finite(r['age']) for r in rows),
                       literal_rules_files_and_time_match=len(candidates),candidate_subjects=len(candidate_by_patient),
                       candidates_with_at_least_two_times=sum(len({r[time_field(cohort)] for r in group})>=2 for group in candidate_by_patient.values()),
                       age_range=[min(float(r['age']) for r in candidates),max(float(r['age']) for r in candidates)],
                       patients_with_literal_CN_and_AD=sum({'CN','AD'} <= {r['class'] for r in group} for group in by_patient.values()),
                       age_missing_rows=sum(not finite(r['age']) for r in rows),time_field=time_field(cohort),
                       scan_time_invalid_rows=[r['uid'] for r in rows if not math.isfinite(ordered_time(cohort,r))])
        if cohort == 'AIBL':
            gaps, invalid_exams = [], []
            for row in rows:
                exam = parse_date(row['date_exam'], '%m/%d/%Y')
                image = parse_date(row['date_image'], '%m/%d/%Y')
                if exam is None:
                    invalid_exams.append(dict(uid=row['uid'],date_exam=row['date_exam']))
                if exam is not None and image is not None:
                    gaps.append((image-exam).days)
            summary['image_minus_exam_days'] = dict(n=len(gaps),min=min(gaps),max=max(gaps),
                absolute_over_180=sum(abs(day)>180 for day in gaps),median=float(np.median(gaps)))
            summary['invalid_exam_dates'] = invalid_exams
            age_steps = []
            for key, group in by_patient.items():
                ordered = sorted(group, key=lambda r: ordered_time(cohort, r))
                for earlier, later in zip(ordered, ordered[1:]):
                    days = ordered_time(cohort, later)-ordered_time(cohort, earlier)
                    if days > 0 and finite(earlier['age']) and finite(later['age']):
                        delta_age = float(later['age'])-float(earlier['age'])
                        age_steps.append(dict(patient_key=key, earlier_uid=earlier['uid'],
                                              later_uid=later['uid'], earlier_age=earlier['age'],
                                              later_age=later['age'], delta_days=int(days),
                                              delta_age=delta_age,
                                              delta_age_minus_elapsed_years=delta_age-days/365.25))
            summary['age_chronology_review'] = dict(
                scope='all metadata; review only, no correction or exclusion',
                adjacent_distinct_date_pairs=len(age_steps),
                age_decreases=sum(p['delta_age']<0 for p in age_steps),
                age_step_differs_from_elapsed_years_by_over_one=sum(abs(p['delta_age_minus_elapsed_years'])>1 for p in age_steps),
                age_decrease_details=[p for p in age_steps if p['delta_age']<0])
        if cohort == 'OASIS2':
            summary['class_CN_but_CDR_positive'] = sum(r['class']=='CN' and finite(r['cdr']) and float(r['cdr'])>0 for r in rows)
            summary['class_CN_but_CDR_positive_subjects'] = len({r['pid'] for r in rows if r['class']=='CN' and finite(r['cdr']) and float(r['cdr'])>0})
            summary['unique_MR_sessions'] = len({oasis2_session(r['uid']) for r in rows})
            summary['nonstandard_scan_suffixes'] = [r['uid'] for r in rows if '_mpr-' not in r['uid']]
            summary['mean_scans_per_MR_session'] = len(rows)/summary['unique_MR_sessions']
        if cohort == 'OASIS3':
            summary['literal_CN_AD_age_missing_rows'] = sum(r['class'] in ('CN','AD') and not finite(r['age']) for r in rows)
            summary['CDR_values_above_three'] = sum(finite(r['cdr']) and float(r['cdr'])>3 for r in rows)
        summaries[cohort] = summary
        chosen = sorted({r['uid'] for r in candidates})
        for index in sorted({0,len(chosen)//2,len(chosen)-1}):
            uid=chosen[index]
            samples.append(dict(cohort=cohort,uid=uid,**sample_mask(masks[uid])))
        print(json.dumps({k:summary[k] for k in ['metadata_rows','literal_CN_AD_rows','finite_age_literal_CN_AD_rows','literal_rules_files_and_time_match','candidate_subjects']},ensure_ascii=False),cohort,flush=True)
    adni_by_patient = defaultdict(list)
    for row in adni_rows:
        adni_by_patient[row['patient_key']].append(row)
    conversions=[]
    for key, group in adni_by_patient.items():
        selected=[r for r in group if r['class'] in ('CN','AD') and finite(r['age'])]
        pairs=[(a,b) for a in selected for b in selected if a['class']=='CN' and b['class']=='AD' and ordered_time(a['cohort'],a)<ordered_time(b['cohort'],b) and float(a['age'])<float(b['age'])]
        if pairs:
            conversions.append(dict(patient_key=key,pids=sorted({r['pid'] for r in group}),cohorts=sorted({r['cohort'] for r in group}),endpoint_pairs=len(pairs),
                                    distinct_date_endpoint_pairs=len({(a['day'],b['day']) for a,b in pairs})))
    adni_by_uid = defaultdict(list)
    for row in adni_rows:
        adni_by_uid[row['uid']].append(row)
    adni_candidates = {key:[r for r in group if r['class'] in ('CN','AD') and finite(r['age'])]
                       for key, group in adni_by_patient.items()}
    overview = dict(purpose='candidate review, not finalized diagnosis mapping or train/test data',
                    proposed_cohorts=COHORTS,cohorts=summaries,
                    metadata_rows=len(records),finite_age_literal_CN_AD_rows=sum(r['literal_rules_match'] for r in records),
                    literal_rules_files_and_time_match=sum(r['literal_rules_files_and_time_match'] for r in records),
                    merged_patient_keys=len(all_by_patient),
                    merged_ADNI_patients=len(adni_by_patient),
                    merged_ADNI_literal_CN_AD_candidate_patients=sum(bool(group) for group in adni_candidates.values()),
                    merged_ADNI_literal_CN_AD_multiple_date_patients=sum(len({r['day'] for r in group})>=2 for group in adni_candidates.values()),
                    ADNI_patients_in_multiple_phases=sum(len({r['cohort'] for r in group})>1 for group in adni_by_patient.values()),
                    ADNI_NC_to_AD_patients=len(conversions),ADNI_NC_to_AD_endpoint_pairs=sum(c['endpoint_pairs'] for c in conversions),
                    ADNI_NC_to_AD_details=conversions,
                    ADNI_NC_to_AD_distinct_date_endpoint_pairs=sum(c['distinct_date_endpoint_pairs'] for c in conversions),
                    ADNI_duplicate_UIDs={uid:[dict(cohort=r['cohort'],pid=r['pid'],age=r['age'],day=r['day'],diagnosis=r['class']) for r in group] for uid,group in adni_by_uid.items() if len(group)>1},
                    other_cohort_identity_links_unverified=True,
                    training_started=False,original_sources_modified=False)
    with (OUTPUT/'all_observations.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (OUTPUT/'summary.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False))
    (OUTPUT/'parcellation_samples.json').write_text(json.dumps(samples,indent=2))
    print(json.dumps({k:overview[k] for k in ['metadata_rows','finite_age_literal_CN_AD_rows','literal_rules_files_and_time_match','merged_ADNI_patients','ADNI_patients_in_multiple_phases','ADNI_NC_to_AD_patients','ADNI_NC_to_AD_endpoint_pairs']},indent=2),flush=True)


if __name__=='__main__':
    main()
