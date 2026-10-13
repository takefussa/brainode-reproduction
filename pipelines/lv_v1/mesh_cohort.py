"""Observation ordering and review flags for cohort mesh generation."""

from datetime import date
import hashlib
import math
import re

QC_THRESHOLDS = dict(min_dice=0.90, max_abs_volume_bias_percent=5.0,
                     max_boundary_mean_mm=1.0)
ORDERING_POLICY = 'ascending ISO scan date, then numeric UID (lexical UID for nonnumeric identifiers); retain every repeat'


def ordered_scans(rows):
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError('no observations for subject')
    if len({row['pid'] for row in rows}) != 1:
        raise ValueError('a fitting sequence must contain one merged patient')
    if len({row['uid'] for row in rows}) != len(rows):
        raise ValueError('UID collision within patient: distinct artifact names required')
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['uid']):
            raise ValueError('UID contains unsafe artifact path characters')
        if date.fromisoformat(row['day']).isoformat() != row['day']:
            raise ValueError('scan dates must be ISO dates')
        if not math.isfinite(float(row['age'])) or row['class'] not in ('CN', 'AD'):
            raise ValueError('finite age and CN/AD required by the baseline table')
    rows.sort(key=lambda row: (row['day'], (0, int(row['uid']), row['uid']) if row['uid'].isdigit()
                              else (1, 0, row['uid'])))
    for a, b in zip(rows, rows[1:]):
        if a['day'] == b['day'] and (float(a['age']), a['class']) != (float(b['age']), b['class']):
            raise ValueError('same-day observations have conflicting age or diagnosis')
        if a['day'] < b['day'] and float(a['age']) > float(b['age']):
            raise ValueError('age decreases with later scan date')
    return rows


def subject_folder(subject):
    readable = re.sub(r'[^A-Za-z0-9_-]', '_', subject)
    return readable + '_' + hashlib.sha256(subject.encode()).hexdigest()[:8]


def review_flags(record):
    """Mark candidates for review; never remove observations from the cohort."""
    after = record['after']
    flags = []
    for key in ('watertight', 'winding_consistent'):
        if not after[key]:
            flags.append(key + '_failed')
    volume = after['volume_mm3']
    if not math.isfinite(volume) or volume <= 0:
        flags.append('nonpositive_or_nonfinite_volume')
    if after['self_intersecting_triangle_pairs']:
        flags.append('self_intersection')
    dice = record['overlap'].get('dice')
    if dice is None or not math.isfinite(dice) or dice < QC_THRESHOLDS['min_dice']:
        flags.append('low_or_unavailable_dice')
    if abs(record['volume_bias_percent']) > QC_THRESHOLDS['max_abs_volume_bias_percent']:
        flags.append('volume_bias')
    if after['boundary_to_surface_mean_mm'] > QC_THRESHOLDS['max_boundary_mean_mm']:
        flags.append('boundary_distance')
    if record['mask_components_26'] != 1:
        flags.append('mask_multiple_components')
    if record['mask_touches_image_edge']:
        flags.append('mask_touches_image_edge')
    return flags
