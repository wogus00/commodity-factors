import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from export_modeling_deployment import package_export
from upload_modeling_blob import inspect_export, publish, main
from export_modeling_bundle import table


@pytest.fixture
def packaged(tmp_path):
    manifest = package_export(ROOT/'data/synthetic/demo', tmp_path)
    return tmp_path, manifest


class Store:
    access = 'public'
    token = 'test-only'
    def __init__(self):
        self.objects = {}; self.calls = []; self.fail_pointer = False
        self.sdk = SimpleNamespace(head=lambda path, **kwargs: SimpleNamespace(url=self.url(path)))
    def url(self, path):
        return 'https://example.public.blob.vercel-storage.com/'+path
    def get(self, path):
        return self.objects.get(path)
    def put(self, path, content, **kwargs):
        self.calls.append(path)
        if path.endswith('latest.json') and self.fail_pointer:
            raise RuntimeError('Upload failed')
        self.objects[path] = content
        return self.url(path)


def approval(manifest):
    return dict(sha256=manifest['sha256'], public_redistribution_approved=True,
                reviewed_by='test', permission_reference='synthetic test fixture only')


def test_packaging_preserves_all_rows_and_precision(packaged):
    directory, manifest = packaged
    first = (directory/manifest['filename']).read_bytes()
    assert package_export(ROOT/'data/synthetic/demo', directory) == manifest
    assert (directory/manifest['filename']).read_bytes() == first
    assert gzip.decompress(first) == (directory/'modeling.json').read_bytes()
    values = [1.2345678901234567, 1e-18, -0.12345678901234567]
    assert [row[0] for row in table(pd.DataFrame({'value': values}))['data']] == values
    assert manifest['table_rows']['states'] == len(pd.read_parquet(ROOT/'data/synthetic/demo/states.parquet'))


def test_upload_dry_run_needs_no_token_or_network(packaged, monkeypatch, capsys):
    directory, _ = packaged
    monkeypatch.delenv('BLOB_READ_WRITE_TOKEN', raising=False)
    monkeypatch.setattr(sys, 'argv', ['upload', '--directory', str(directory)])
    main()
    assert '"upload": false' in capsys.readouterr().out


def test_approval_and_private_store_block_writes(packaged):
    directory, manifest = packaged
    store = Store()
    with pytest.raises(ValueError, match='approval'):
        publish(directory, {}, store)
    store.access = 'private'
    with pytest.raises(ValueError, match='public store'):
        publish(directory, approval(manifest), store)
    assert store.calls == []


def test_manifest_committed_last_and_retry_reuses_version(packaged):
    directory, manifest = packaged
    store = Store();store.fail_pointer = True
    with pytest.raises(RuntimeError):
        publish(directory, approval(manifest), store)
    assert len(store.objects) == 1
    store.fail_pointer = False
    url = publish(directory, approval(manifest), store)
    assert url.endswith('/latest.json')
    assert len([path for path in store.calls if '/versions/' in path]) == 1
    pointer = json.loads(store.objects[store.calls[-1]])
    assert pointer['sha256'] == manifest['sha256']
    assert 'permission_reference' not in pointer


def test_tampered_exports_fail_before_upload(packaged):
    directory, manifest = packaged
    (directory/manifest['filename']).write_bytes(b'corrupted')
    store = Store()
    with pytest.raises(ValueError, match='checksum'):
        publish(directory, approval(manifest), store)
    assert store.calls == []
