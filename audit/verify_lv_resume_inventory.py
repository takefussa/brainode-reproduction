"""Verify saved LV prefixes and source hashes before a host resume."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Inventory exists; use a new output path')
    config = json.loads(args.config.read_text())
    sys.path.insert(0, str(args.snapshot.resolve()))
    import numpy as np
    from pipeline import ordered, paths
    from mesh_cohort import subject_folder
    from mesh_initialization import initialize_template
    groups = ordered(config)
    missing = [row['parcellation_path'] for rows in groups.values() for row in rows
               if not Path(row['parcellation_path']).is_file()]
    if missing:
        raise RuntimeError('Missing source file: ' + missing[0])
    manifests, sources = [], {}
    verified_artifacts = 0
    for side in ('left', 'right'):
        with np.load(args.snapshot / 'assets' / f'lv_{side}_template.npz') as data:
            template, faces = data['vertices'].copy(), data['faces'].copy()
        for pid, rows in groups.items():
            directory = paths(config, side)['mesh'] / subject_folder(pid)
            path = directory / 'manifest.json'
            if not path.exists():
                continue
            before = digest(path)
            manifest = json.loads(path.read_text())
            records = manifest['scans']
            if manifest['subject'] != pid or len(records) > len(rows):
                raise ValueError('Cached patient prefix mismatch: ' + str(path))
            if manifest['iterations'] != config['mesh_iterations'] or manifest['labels'] != config['regions'][side]:
                raise ValueError('Cached fitting settings mismatch: ' + str(path))
            if manifest['template_sha256'] != digest(args.snapshot / 'assets' / f'lv_{side}_template.npz'):
                raise ValueError('Cached template mismatch: ' + str(path))
            previous = None
            if records:
                previous, _ = initialize_template(template, faces, mode='mask-volume',
                                                  mask_volume=records[0]['mask_volume_mm3'])
            for record, row in zip(records, rows):
                if (record['uid'], record['day'], record['age'], record['diagnosis']) != (
                        row['uid'], row['day'], float(row['age']), row['class']):
                    raise ValueError('Cached chronological prefix mismatch: ' + str(path))
                source = row['parcellation_path']
                if source in sources and sources[source] != record['source_sha256']:
                    raise ValueError('Conflicting source hashes: ' + source)
                sources[source] = record['source_sha256']
                expected = {record['uid'] + suffix for suffix in ('.npz', '.obj', '_loss.csv')}
                if set(record['artifact_sha256']) != expected:
                    raise ValueError('Incomplete cached artifact set: ' + str(path))
                for name, sha in record['artifact_sha256'].items():
                    if Path(name).name != name or digest(directory / name) != sha:
                        raise ValueError('Changed cached artifact: ' + str(directory / name))
                    verified_artifacts += 1
                with np.load(directory / (record['uid'] + '.npz'), allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['faces'], faces)
                    np.testing.assert_array_equal(data['source_vertices'], previous)
                    previous = data['vertices'].copy()
                    if previous.shape != (2490, 3) or not np.isfinite(previous).all():
                        raise ValueError('Invalid cached LV vertices')
            if digest(path) != before:
                raise RuntimeError('Manifest changed during verification: ' + str(path))
            manifests.append(dict(side=side, subject=pid, manifest=str(path), sha256=before,
                                  completed_scans=len(records), environment=manifest['environment']))
        print('Verified mesh artifacts:', side, sum(m['completed_scans'] for m in manifests if m['side'] == side), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(digest, path): (path, expected) for path, expected in sources.items()}
        for count, future in enumerate(as_completed(futures), 1):
            path, expected = futures[future]
            if future.result() != expected:
                raise ValueError('Source segmentation changed: ' + path)
            if count % 100 == 0 or count == len(futures):
                print(f'Verified source hashes: {count}/{len(futures)}', flush=True)
    for record in manifests:
        if digest(record['manifest']) != record['sha256']:
            raise RuntimeError('Manifest changed before inventory commit')
    state = dict(status='verified', completed_meshes=sum(m['completed_scans'] for m in manifests),
                 manifests=manifests, artifact_files_verified=verified_artifacts,
                 source_hashes_verified=len(sources), eligible_sources_accessible=config['scans'],
                 verified_utc=datetime.now(timezone.utc).isoformat(),
                 config_sha256=digest(args.config), snapshot=str(args.snapshot.resolve()),
                 resume_boundary='completed scan prefixes; unfinished fits restart at update 1',
                 previous_inventory=str(args.output.parent / 'migration_inventory.json'))
    args.output.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in state.items() if k != 'manifests'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
