"""Session-private temporary frames, deliberately outside the history CAS."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import uuid


_HANDLE = re.compile(r"memory:([0-9a-f]{64}):([0-9a-f]{64})\Z")


class SessionFrames:
    def __init__(self, db_path):
        self.root = Path(str(db_path) + '.frames')

    def _directory(self, session_id):
        return self.root / hashlib.sha256(session_id.encode('utf-8')).hexdigest()

    def put(self, session_id, data):
        directory = self._directory(session_id)
        if self.root.is_symlink() or directory.is_symlink():
            raise ValueError('memory frame directory must not be a symlink')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.mkdir(mode=0o700, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        target = directory / (digest + '.jpg')
        part = directory / (uuid.uuid4().hex + '.part')
        try:
            with os.fdopen(os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
                stream.write(data)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
        return f'memory:{directory.name}:{digest}'

    def load(self, session_id, handle):
        match = _HANDLE.fullmatch(handle or '')
        directory = self._directory(session_id)
        if not match or match[1] != directory.name:
            raise ValueError('frame does not belong to this session')
        path = directory / (match[2] + '.jpg')
        if self.root.is_symlink() or directory.is_symlink() or path.is_symlink():
            raise ValueError('memory frame path must not be a symlink')
        return path.read_bytes()

    def delete_session(self, session_id):
        directory = self._directory(session_id)
        if self.root.is_symlink() or directory.is_symlink():
            raise ValueError('memory frame directory must not be a symlink')
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
