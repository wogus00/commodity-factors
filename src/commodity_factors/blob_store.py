"""Private recoverable model history and public versioned dashboard results."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

from commodity_factors.analysis import TABLES, load_bundle, save_bundle
from commodity_factors.live_feed import DATASET_ID
from commodity_factors.web_bundle import browser_bytes


PREFIX = "commodity-factors/wti"
EXPECTED_FILES = {f"{name}.parquet" for name in TABLES} | {"metadata.json"}


def pack_bundle(bundle):
    output = BytesIO()
    with TemporaryDirectory() as directory:
        save_bundle(bundle, directory)
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            for name in sorted(EXPECTED_FILES):
                archive.write(Path(directory) / name, arcname=name)
    return output.getvalue()


def unpack_bundle(content):
    with TemporaryDirectory() as directory, ZipFile(BytesIO(content)) as archive:
        entries = archive.infolist()
        if (len(entries) != len(EXPECTED_FILES) or {e.filename for e in entries} != EXPECTED_FILES
                or sum(e.file_size for e in entries) > 200_000_000):
            raise ValueError("Invalid or oversized saved model archive")
        for entry in entries:
            # Only the six known filenames are written; no archive paths or symlinks are used.
            (Path(directory) / entry.filename).write_bytes(archive.read(entry))
        return load_bundle(directory)


class BlobStore:
    def __init__(self, token, access, *, sdk=None):
        if not token or access not in {"private", "public"}:
            raise ValueError("A Blob token and explicit store access are required")
        if sdk is None:
            from vercel import blob as sdk
        self.sdk, self.token, self.access = sdk, token, access

    def get(self, path):
        try:
            result = self.sdk.get(path, token=self.token, access=self.access, use_cache=False, timeout=60)
        except self.sdk.BlobNotFoundError:
            return None
        if result.status_code != 200:
            raise RuntimeError("Unable to read the saved Blob object")
        return result.content

    def put(self, path, content, *, content_type, overwrite=False):
        result = self.sdk.put(path, content, access=self.access, token=self.token,
                              content_type=content_type, overwrite=overwrite,
                              add_random_suffix=False, cache_control_max_age=60 if overwrite else 31536000)
        expected = f".{self.access}.blob.vercel-storage.com"
        from urllib.parse import urlparse
        if not (urlparse(result.url).hostname or "").endswith(expected):
            raise RuntimeError("Blob store access does not match the requested privacy level")
        return result.url

    def prune(self, prefix, *, keep=7, protected=()):
        cursor, items = None, []
        while True:
            page = self.sdk.list_objects(prefix=prefix, cursor=cursor, token=self.token, limit=1000)
            items.extend(page.blobs)
            if not page.has_more:
                break
            cursor = page.cursor
        ordered = sorted(items, key=lambda item: item.uploaded_at, reverse=True)
        retained = set(protected) | {item.url for item in ordered[:keep]}
        expired = [item.url for item in ordered if item.url not in retained]
        if expired:
            self.sdk.delete(expired, token=self.token)


class LiveRepository:
    def __init__(self, private, public):
        if private.access != "private" or public.access != "public":
            raise ValueError("Model history requires a private store; dashboard data requires a public store")
        self.private, self.public = private, public

    def load(self):
        content = self.private.get(f"{PREFIX}/latest.json")
        if content is None:
            return None
        pointer = json.loads(content)
        archive = self.private.get(pointer["archive_url"])
        if archive is None:
            raise RuntimeError("Saved model archive is missing; initialization cannot replace it")
        bundle = unpack_bundle(archive)
        if bundle.metadata.get("dataset_id") != DATASET_ID:
            raise ValueError("Automatic WTI history must come from Databento, separately from WRDS")
        return bundle

    def publish(self, bundle, *, display_approved):
        if not display_approved or bundle.metadata.get("dataset_id") != DATASET_ID:
            raise ValueError("Public data requires approved Databento display rights")
        version = uuid4().hex
        archive_url = self.private.put(f"{PREFIX}/versions/{version}.zip", pack_bundle(bundle),
                                       content_type="application/zip")
        data_url = self.public.put(f"{PREFIX}/versions/{version}.json", browser_bytes(bundle),
                                  content_type="application/json")
        manifest = {
            "format": "commodity-factors-live", "version": 1,
            "dataset_id": DATASET_ID, "data_url": data_url,
            "as_of": str(bundle.panel.date_.max().date()),
            "updated_at": bundle.metadata["daily_forecast"]["last_update_utc"],
            "next_session": bundle.metadata["daily_forecast"]["next_session"],
        }
        # Upload immutable objects first. Commit recoverable model state before
        # exposing its data URL. A failed public pointer update is safely retried.
        self.private.put(f"{PREFIX}/latest.json", json.dumps(manifest | {"archive_url": archive_url}).encode(),
                         content_type="application/json", overwrite=True)
        manifest_url = self.public.put(f"{PREFIX}/latest.json", json.dumps(manifest).encode(),
                                       content_type="application/json", overwrite=True)
        self.private.prune(f"{PREFIX}/versions/", protected=(archive_url,))
        self.public.prune(f"{PREFIX}/versions/", protected=(data_url,))
        return manifest_url
