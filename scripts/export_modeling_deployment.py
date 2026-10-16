#!/usr/bin/env python3
"""Package all saved modeling rows for Blob; no uploads or model fitting."""
import argparse
import gzip
import hashlib
import json
from io import BytesIO
from pathlib import Path
from export_modeling_bundle import export_modeling_bundle


def package_export(bundle, output_dir, pca=None, robustness=None, relaxed=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = export_modeling_bundle(bundle, output_dir/'modeling.json', pca, robustness, relaxed)
    raw = path.read_bytes()
    payload = json.loads(raw)
    compressed = BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb', filename='', mtime=0) as stream:
        stream.write(raw)
    content = compressed.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    filename = f'{digest}.json.gz'
    (output_dir/filename).write_bytes(content)
    panel = payload['tables']['panel']
    dates = [row[panel['columns'].index('date_')][:10] for row in panel['data']]
    manifest = dict(format='wti-modeling-deployment', version=1, encoding='gzip',
                    dataset_id=payload['metadata']['dataset_id'], filename=filename,
                    sha256=digest, bytes=len(content), decoded_bytes=len(raw),
                    first_date=min(dates), last_date=max(dates),
                    table_rows={name:len(table['data']) for name,table in payload['tables'].items()})
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('results/wti'))
    parser.add_argument('--pca', type=Path, default=Path('results/pca/wti'))
    parser.add_argument('--robustness', type=Path, default=Path('results/pca/wti/robustness'))
    parser.add_argument('--relaxed-sensitivity', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('exports/wti'))
    args = parser.parse_args()
    print(json.dumps(package_export(args.bundle, args.output_dir, args.pca, args.robustness, args.relaxed_sensitivity), indent=2))
