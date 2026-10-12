"""Content identities for private experiment caches with changing image/detector inputs."""
import hashlib
from pathlib import Path


def files_digest(paths):
    digest=hashlib.sha256()
    for path in sorted({Path(p).resolve() for p in paths}):
        digest.update(str(path).encode());digest.update(b'\0')
        file_hash=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):file_hash.update(chunk)
        digest.update(file_hash.digest())
    return digest.hexdigest()
