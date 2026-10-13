"""Prepare the ADNI baseline split; no mesh/PCA/model training."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab_dataset import AgeTimeScale, eligible_visits, normalized_visits, write_table

PROJECT=Path(__file__).resolve().parents[1]


def main():
    source=PROJECT/'artifacts/multicohort_inventory/all_observations.csv'
    with source.open() as file:
        raw=list(csv.DictReader(file))
    rows=[]
    for r in raw:
        if r['cohort'] not in ('ADNI1','ADNI2','ADNI3'):
            continue
        rows.append(dict(pid=r['patient_key'],original_pid=r['pid'],cohort=r['cohort'],
            uid=r['uid'],scan_key=r['scan_key'],age=r['age'],day=r['time_value'],
            **{'class':r['diagnosis']},parcellation_path=r['parcellation_path'],
            trsaa_path=r['trsaa_path'],raw_metadata=r['raw_metadata']))
    accepted,rejected=eligible_visits(rows)
    subjects=sorted({r['pid'] for r in accepted})
    permutation=np.random.default_rng(42).permutation(subjects)
    train=set(permutation[:int(len(subjects)*.8)])
    test=set(subjects)-train
    assert not train&test and train|test==set(subjects)
    scale=AgeTimeScale.fit(accepted,train_subjects=train)
    accepted=normalized_visits(accepted,scale)
    for r in accepted:
        r['split']='train' if r['pid'] in train else 'test'
        if not Path(r['parcellation_path']).is_file() or not Path(r['trsaa_path']).is_file():
            raise ValueError('required source file missing')
    for r in rejected:
        r['split']='unassigned'
    output=PROJECT/'artifacts/adni_all_baseline'
    output.mkdir(parents=True,exist_ok=False)
    fields=list(rows[0])
    write_table(output/'eligible.csv',accepted,fields+['observed_c','time','split'])
    write_table(output/'excluded.csv',rejected,fields+['exclusion_reasons','split'])
    (output/'train_subjects.txt').write_text('\n'.join(sorted(train))+'\n')
    (output/'test_subjects.txt').write_text('\n'.join(sorted(test))+'\n')
    repeat=defaultdict(list)
    patients=defaultdict(list)
    for r in accepted:
        repeat[(r['pid'],r['day'])].append(r)
        patients[r['pid']].append(r)
    age_reversals=[]
    for pid,group in patients.items():
        ordered=sorted(group,key=lambda r:(r['day'],r['uid']))
        for a,b in zip(ordered,ordered[1:]):
            if a['day']<b['day'] and float(a['age'])>float(b['age']):
                age_reversals.append(dict(pid=pid,earlier_uid=a['uid'],later_uid=b['uid'],
                    earlier_age=a['age'],later_age=b['age'],earlier_day=a['day'],later_day=b['day']))
    conversions={split:[] for split in ('train','test')}
    for pid,group in patients.items():
        if any(a['class']=='CN' and b['class']=='AD' and a['day']<b['day']
               and float(a['age'])<float(b['age']) for a in group for b in group):
            conversions['train' if pid in train else 'test'].append(pid)
    parameters=dict(min_age=scale.min_age,max_age=scale.max_age,
        span_years=scale.max_age-scale.min_age,train_subjects=sorted(train),
        train_scan_keys=sorted(r['scan_key'] for r in accepted if r['split']=='train'),
        fit_scope='eligible training subjects only',test_and_inference='reuse without refitting or clipping')
    (output/'time_scale.json').write_text(json.dumps(parameters,indent=2))
    summary=dict(date='2026-10-07 Asia/Tokyo',
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_rows=len(rows),eligible_scans=len(accepted),excluded_scans=len(rejected),
        eligible_subjects=len(subjects),split_subjects=dict(train=len(train),test=len(test)),
        split_scans=dict(Counter(r['split'] for r in accepted)),
        class_counts_by_split={s:dict(Counter(r['class'] for r in accepted if r['split']==s)) for s in ('train','test')},
        split_algorithm='sort merged RID keys, NumPy default_rng(42) permutation, first floor(0.8*N) train',
        seed=42,age_range=[min(float(r['age']) for r in accepted),max(float(r['age']) for r in accepted)],
        train_time_scale=dict(min_age=scale.min_age,max_age=scale.max_age,span_years=scale.max_age-scale.min_age),
        same_day_repeat_groups=sum(len(g)>1 for g in repeat.values()),
        same_day_repeat_policy='retain every row as a separate observation at its unchanged time',
        same_day_age_or_class_conflicts=[dict(pid=p,day=d,uids=[r['uid'] for r in g])
            for (p,d),g in repeat.items() if len({r['age'] for r in g})>1 or len({r['class'] for r in g})>1],
        age_decreases_with_later_date=age_reversals,
        NC_to_AD_subjects={s:len(p) for s,p in conversions.items()},
        NC_to_AD_subject_keys=conversions,
        PCA_plan='150 components fitted on all eligible training observations; not fitted yet',
        mesh_fit_quality_status='not reviewed across cohort; no mesh accepted for training by this script',
        training_started=False,original_sources_modified=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ['source_rows','eligible_scans','eligible_subjects','split_subjects',
        'split_scans','train_time_scale','same_day_repeat_groups','NC_to_AD_subjects']},indent=2))
    print('same-day age/class conflicts',len(summary['same_day_age_or_class_conflicts']))
    print('age reversals',len(age_reversals))


if __name__=='__main__':
    main()
