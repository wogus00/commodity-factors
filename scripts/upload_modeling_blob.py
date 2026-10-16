#!/usr/bin/env python3
"""Dry-run by default. Publish reviewed modeling exports to a PUBLIC Blob store."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

PREFIX = 'commodity-factors/modeling/wti'


def inspect_export(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    digest = manifest.get('sha256', '')
    if (manifest.get('format') != 'wti-modeling-deployment' or manifest.get('version') != 1
            or manifest.get('encoding') != 'gzip' or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or manifest.get('filename') != f'{digest}.json.gz'):
        raise ValueError('Invalid deployment manifest')
    content = (directory/manifest['filename']).read_bytes()
    if len(content) > 30_000_000 or len(content) != manifest['bytes'] or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('Export size or checksum mismatch')
    from io import BytesIO
    with gzip.GzipFile(fileobj=BytesIO(content)) as stream:
        raw = stream.read(30_000_001)
    if len(raw) > 30_000_000 or len(raw) != manifest['decoded_bytes']:
        raise ValueError('Decoded export exceeds browser limit or differs from manifest')
    payload = json.loads(raw)
    if (payload.get('format') != 'crude-oil-factor-modeling' or payload.get('version') != 1
            or payload['metadata']['dataset_id'] != manifest['dataset_id']
            or {key:len(value['data']) for key,value in payload['tables'].items()} != manifest['table_rows']):
        raise ValueError('Modeling payload does not match manifest')
    return manifest, content


def publish(directory, approval, store):
    manifest, content = inspect_export(directory)
    if (approval.get('sha256') != manifest['sha256'] or approval.get('public_redistribution_approved') is not True
            or not str(approval.get('permission_reference', '')).strip()
            or not str(approval.get('reviewed_by', '')).strip()):
        raise ValueError('Public redistribution approval must name a reviewer, permission reference and this exact SHA256')
    if store.access != 'public':
        raise ValueError('This direct-browser pipeline requires a public store; private access needs authentication')
    path = f"{PREFIX}/versions/{manifest['filename']}"
    # Reuse identical immutable objects on retries. Never prune or delete data.
    existing = store.get(path)
    if existing is not None and existing != content:
        raise ValueError('Existing version has unexpected contents')
    if existing is None:
        url = store.put(path, content, content_type='application/gzip')
    else:
        # A head avoids overwriting immutable objects and returns the store URL.
        url = store.sdk.head(path, token=store.token).url
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.public.blob.vercel-storage.com'):
        raise ValueError('Expected a public Blob URL')
    remote_manifest = manifest | {'data_url':url}
    result = store.put(f'{PREFIX}/latest.json', json.dumps(remote_manifest, separators=(',', ':')).encode(),
                      content_type='application/json', overwrite=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('exports/wti'))
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest, _ = inspect_export(args.directory)
    size = manifest['bytes']/1e9
    print(json.dumps(dict(manifest=manifest, upload=False, illustrative_usd={
        'one_version_month_storage_at_0_023_per_GB':size*0.023,
        '1000_full_downloads_at_0_05_per_GB':size*1000*0.05},
        cost_note='Excludes plan fees, CDN requests, cache misses, operations and retained versions; verify current regional pricing.'), indent=2))
    if not args.execute:
        return
    if not args.approval:
        parser.error('--execute requires --approval; dry-run performs no network requests')
    approval = json.loads(args.approval.read_text())
    token = os.environ.get('BLOB_READ_WRITE_TOKEN')
    if not token:
        parser.error('Set BLOB_READ_WRITE_TOKEN in the local process environment, never browser configuration')
    from commodity_factors.blob_store import BlobStore
    print('MODELING_MANIFEST_URL='+publish(args.directory, approval, BlobStore(token, 'public')))


if __name__ == '__main__':
    main()
