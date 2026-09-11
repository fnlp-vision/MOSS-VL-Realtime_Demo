"""Memory store: rows in SQLite, vectors as float32 blobs + a numpy brute force.

Why not an ANN index: at session scale (hundreds to low thousands of items) an
exact scan of a (N, D) float32 matrix is well under a millisecond, while every
ANN structure imports write-path problems we would have to babysit — no
deletes, fragment proliferation, periodic rebuilds. Brute force is also exact,
so there is no recall knob to tune. sqlite-vec would only become interesting
past ~100k vectors per conversation, which a live session never reaches.

Why a separate `memory.db` rather than tables inside `index.db`: the history
index is a *derived projection* that `scripts/history_prune.py --rebuild` drops
and regenerates from the journals. Memory rows are not reconstructible that way
(they carry embeddings and LLM-written captions), so putting them in index.db
would make them collateral damage of a rebuild.

Isolation: every read and write is scoped by `conversation_id`, and the in-RAM
vector matrices are per (conversation, space). A session can only ever see its
own memories; there is no cross-session query path in this module.

Threading: one write connection under a lock (the writer thread owns it); reads
open short-lived WAL connections. Nothing here may be called on the event loop
— retrieval hops through asyncio.to_thread.
"""
from __future__ import annotations

import os
import fcntl
import shutil
import sqlite3
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

KIND_UTTERANCE = "utterance"
KIND_CAPTION = "caption"
KIND_FRAME = "frame"
KIND_FACT = "fact"
KIND_PINNED = "pinned"

SPACE_TEXT = "text"
SPACE_IMAGE = "image"
# late-interaction token matrices (BGE-M3 colbert head). Unlike the pooled
# spaces these are VARIABLE-length (T, dim) rows, so they live outside the
# fixed-dim _VecIndex scan: the `dim` column carries the token width and the
# blob is self-describing — a 4-byte little-endian row-count header followed
# by row-major float32 of shape (rows, dim). See _pack_li/_unpack_li.
SPACE_TEXT_LI = "text_li"

_LI_HEADER = struct.Struct("<I")


def _pack_li(mat: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _LI_HEADER.pack(arr.shape[0]) + arr.tobytes()


def _unpack_li(blob: bytes, dim: int) -> np.ndarray:
    (rows,) = _LI_HEADER.unpack_from(blob, 0)
    return np.frombuffer(blob, dtype=np.float32, offset=_LI_HEADER.size).reshape(rows, dim)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
  id              INTEGER PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  kind            TEXT NOT NULL,
  role            TEXT,
  text            TEXT,
  lang            TEXT,
  session_ts      REAL,
  media_ts        REAL,
  media_hash      TEXT,
  importance      REAL NOT NULL DEFAULT 0.5,
  valid_from      REAL,
  invalid_at      REAL,
  created_at      REAL NOT NULL,
  source_ids      TEXT,
  injected_in     TEXT
);
CREATE INDEX IF NOT EXISTS memory_items_by_conv ON memory_items(conversation_id, id);

CREATE TABLE IF NOT EXISTS memory_vectors (
  item_id INTEGER NOT NULL,
  space   TEXT NOT NULL,
  dim     INTEGER NOT NULL,
  vec     BLOB NOT NULL,
  PRIMARY KEY (item_id, space)
);

-- retrieval index keys (turn_text + extracted facts) for audit only: the model
-- never sees them, memory_items.text stays the raw verbatim turn (design §3)
CREATE TABLE IF NOT EXISTS memory_item_keys (
  item_id INTEGER PRIMARY KEY,
  key     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_sessions (
  conversation_id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at REAL NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT
);
"""


@dataclass
class MemoryItem:
    id: int
    conversation_id: str
    kind: str
    text: str = ""
    role: Optional[str] = None
    lang: Optional[str] = None
    session_ts: Optional[float] = None
    media_ts: Optional[float] = None
    media_hash: Optional[str] = None
    importance: float = 0.5
    created_at: float = 0.0
    invalid_at: Optional[float] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MemoryItem":
        return cls(
            id=row["id"], conversation_id=row["conversation_id"], kind=row["kind"],
            text=row["text"] or "", role=row["role"], lang=row["lang"],
            session_ts=row["session_ts"], media_ts=row["media_ts"],
            media_hash=row["media_hash"], importance=row["importance"],
            created_at=row["created_at"], invalid_at=row["invalid_at"],
        )


@dataclass
class _VecIndex:
    """Per (conversation, space) vectors held in RAM for the exact scan."""
    ids: List[int] = field(default_factory=list)
    rows: List[np.ndarray] = field(default_factory=list)
    _mat: Optional[np.ndarray] = None
    loaded: bool = False

    def append(self, item_id: int, vec: np.ndarray) -> None:
        self.ids.append(item_id)
        self.rows.append(np.asarray(vec, dtype=np.float32).ravel())
        self._mat = None

    def matrix(self) -> Optional[np.ndarray]:
        if not self.rows:
            return None
        if self._mat is None:
            self._mat = np.vstack(self.rows)
        return self._mat


class MemoryStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        path = (settings.memory_db_path or "").strip()
        self.path = os.path.realpath(path or os.path.join(settings.data_dir, "memory.db"))
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._idx: Dict[Tuple[str, str], _VecIndex] = {}
        # late-interaction matrices: conv -> {item_id: (T, dim) float32}
        self._li: Dict[str, Dict[int, np.ndarray]] = {}
        self._li_loaded: Set[str] = set()
        self.owner_id = uuid.uuid4().hex
        self._file_lock = None
        self._cleanup_lock = threading.Lock()
        self.frames = None
        self.accepting_writes = True
        self._storage_status = {}

    # ---- lifecycle ----

    def open(self) -> None:
        with self._lock:
            if self._conn is not None:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            lock = open(self.path + '.lock', 'a+b')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock.close()
                raise RuntimeError('memory database is owned by another API or maintenance process')
            self._file_lock = lock
            conn = None
            try:
                conn = sqlite3.connect(self.path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.executescript(_SCHEMA)
                conn.commit()
            except Exception:
                if conn is not None:
                    conn.close()
                lock.close()
                self._file_lock = None
                raise
            self._conn = conn
            log.info("memory store open: %s", self.path)

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
            try:
                if conn is not None:
                    conn.commit()
            finally:
                try:
                    if conn is not None:
                        conn.close()
                finally:
                    self._idx.clear()
                    self._li.clear()
                    self._li_loaded.clear()
                    if self._file_lock is not None:
                        self._file_lock.close()
                        self._file_lock = None

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    # ---- writes ----

    def add_item(self, conversation_id: str, kind: str, *, text: str = "",
                 role: Optional[str] = None, lang: Optional[str] = None,
                 session_ts: Optional[float] = None, media_ts: Optional[float] = None,
                 media_hash: Optional[str] = None, importance: float = 0.5,
                 source_ids: Optional[str] = None) -> int:
        now = time.time()
        with self._lock:
            conn = self._require()
            self._register_session(conn, conversation_id)
            cur = conn.execute(
                "INSERT INTO memory_items (conversation_id, kind, role, text, lang, session_ts,"
                " media_ts, media_hash, importance, valid_from, created_at, source_ids)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (conversation_id, kind, role, text, lang, session_ts, media_ts, media_hash,
                 float(importance), now, now, source_ids))
            conn.commit()
            return int(cur.lastrowid)

    def add_vector(self, conversation_id: str, item_id: int, space: str, vec: np.ndarray) -> None:
        arr = np.asarray(vec, dtype=np.float32).ravel()
        with self._lock:
            conn = self._require()
            if not self._owns_item(conn, conversation_id, item_id):
                return
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (item_id, space, dim, vec) VALUES (?,?,?,?)",
                (item_id, space, int(arr.size), arr.tobytes()))
            conn.commit()
            idx = self._idx.get((conversation_id, space))
            if idx is not None and idx.loaded:
                idx.append(item_id, arr)

    def add_vector_late(self, conversation_id: str, item_id: int, mat: np.ndarray) -> None:
        """Store a (T, dim) token matrix in SPACE_TEXT_LI (variable-length blob)."""
        arr = np.ascontiguousarray(mat, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        blob = _pack_li(arr)
        with self._lock:
            conn = self._require()
            if not self._owns_item(conn, conversation_id, item_id):
                return
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (item_id, space, dim, vec) VALUES (?,?,?,?)",
                (item_id, SPACE_TEXT_LI, int(arr.shape[-1]), blob))
            conn.commit()
            if conversation_id in self._li_loaded:
                self._li.setdefault(conversation_id, {})[int(item_id)] = arr

    def update_vector(self, conversation_id: str, item_id: int, space: str, vec: np.ndarray) -> None:
        """Replace an item's vector IN PLACE (fact re-keying, design §3).

        `add_vector` on an existing id would append a duplicate row to the
        in-RAM scan matrix while the blob upsert kept only the newest — the
        index would then score the stale vector forever. Here the loaded index
        row is patched positionally instead.
        """
        arr = np.asarray(vec, dtype=np.float32)
        with self._lock:
            conn = self._require()
            if not self._owns_item(conn, conversation_id, item_id):
                return
            if space == SPACE_TEXT_LI:
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                conn.execute(
                    "INSERT OR REPLACE INTO memory_vectors (item_id, space, dim, vec) VALUES (?,?,?,?)",
                    (item_id, space, int(arr.shape[-1]), _pack_li(arr)))
                conn.commit()
                if conversation_id in self._li_loaded:
                    self._li.setdefault(conversation_id, {})[int(item_id)] = np.ascontiguousarray(arr)
                return
            arr = arr.ravel()
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (item_id, space, dim, vec) VALUES (?,?,?,?)",
                (item_id, space, int(arr.size), arr.tobytes()))
            conn.commit()
            idx = self._idx.get((conversation_id, space))
            if idx is not None and idx.loaded:
                if item_id in idx.ids:
                    pos = idx.ids.index(item_id)
                    idx.rows[pos] = arr
                    idx._mat = None
                else:
                    idx.append(item_id, arr)

    def put_key(self, item_id: int, key: str, *, conversation_id: Optional[str] = None) -> None:
        """Persist the retrieval index key for audit (never shown to the model)."""
        with self._lock:
            conn = self._require()
            row = conn.execute("SELECT conversation_id FROM memory_items WHERE id = ?", (item_id,)).fetchone()
            if row is None or (conversation_id is not None and row["conversation_id"] != conversation_id):
                return
            conn.execute("INSERT OR REPLACE INTO memory_item_keys (item_id, key) VALUES (?,?)",
                         (int(item_id), key))
            conn.commit()

    def get_key(self, item_id: int) -> Optional[str]:
        with self._lock:
            row = self._require().execute(
                "SELECT key FROM memory_item_keys WHERE item_id = ?", (int(item_id),)).fetchone()
        return str(row["key"]) if row else None

    def mark_injected(self, item_id: int, session_id: str) -> None:
        with self._lock:
            conn = self._require()
            conn.execute(
                "UPDATE memory_items SET injected_in = COALESCE(injected_in || ',', '') || ?"
                " WHERE id = ? AND conversation_id = ?", (session_id, item_id, session_id))
            conn.commit()

    def invalidate(self, item_id: int) -> None:
        """Supersede rather than delete (bi-temporal): the row stays auditable."""
        with self._lock:
            conn = self._require()
            conn.execute("UPDATE memory_items SET invalid_at = ? WHERE id = ?", (time.time(), item_id))
            conn.commit()

    # ---- reads ----

    def _load_index(self, conversation_id: str, space: str) -> _VecIndex:
        key = (conversation_id, space)
        idx = self._idx.get(key)
        if idx is not None and idx.loaded:
            return idx
        idx = _VecIndex()
        conn = self._require()
        rows = conn.execute(
            "SELECT v.item_id AS item_id, v.vec AS vec FROM memory_vectors v"
            " JOIN memory_items i ON i.id = v.item_id"
            " WHERE i.conversation_id = ? AND v.space = ? AND i.invalid_at IS NULL"
            " ORDER BY v.item_id", (conversation_id, space)).fetchall()
        for row in rows:
            idx.append(int(row["item_id"]), np.frombuffer(row["vec"], dtype=np.float32))
        idx.loaded = True
        self._idx[key] = idx
        return idx

    def search(self, conversation_id: str, space: str, query: np.ndarray, limit: int = 16,
               exclude: Optional[Iterable[int]] = None) -> List[Tuple[int, float]]:
        """Exact cosine top-k within ONE conversation. Vectors are L2-normalized."""
        q = np.asarray(query, dtype=np.float32).ravel()
        with self._lock:
            idx = self._load_index(conversation_id, space)
            mat = idx.matrix()
            if mat is None or mat.shape[1] != q.size:
                return []
            scores = mat @ q
            ids = idx.ids
        skip = set(exclude or ())
        order = np.argsort(-scores)[: max(limit * 4, limit)]
        out: List[Tuple[int, float]] = []
        for pos in order:
            item_id = ids[int(pos)]
            if item_id in skip:
                continue
            out.append((item_id, float(scores[int(pos)])))
            if len(out) >= limit:
                break
        return out

    def _load_late(self, conversation_id: str) -> Dict[int, np.ndarray]:
        if conversation_id in self._li_loaded:
            return self._li.get(conversation_id, {})
        mats: Dict[int, np.ndarray] = {}
        conn = self._require()
        rows = conn.execute(
            "SELECT v.item_id AS item_id, v.dim AS dim, v.vec AS vec FROM memory_vectors v"
            " JOIN memory_items i ON i.id = v.item_id"
            " WHERE i.conversation_id = ? AND v.space = ? AND i.invalid_at IS NULL"
            " ORDER BY v.item_id", (conversation_id, SPACE_TEXT_LI)).fetchall()
        for row in rows:
            try:
                mats[int(row["item_id"])] = _unpack_li(row["vec"], int(row["dim"]))
            except Exception:  # noqa: BLE001 — one corrupt blob must not mute the lane
                continue
        self._li[conversation_id] = mats
        self._li_loaded.add(conversation_id)
        return mats

    def search_late(self, conversation_id: str, query_tokens: np.ndarray, limit: int = 16,
                    exclude: Optional[Iterable[int]] = None) -> List[Tuple[int, float]]:
        """ColBERT-style max-sim: for each QUERY token, its best cosine over the
        item's token matrix, averaged. Exact loop — fine at session scale, and
        the matrices are far too ragged for the fixed-dim _VecIndex scan."""
        q = np.asarray(query_tokens, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        with self._lock:
            mats = dict(self._load_late(conversation_id))
        skip = set(exclude or ())
        scored: List[Tuple[int, float]] = []
        for item_id, mat in mats.items():
            if item_id in skip or mat.size == 0 or mat.shape[1] != q.shape[1]:
                continue
            sims = mat @ q.T  # (item tokens, query tokens)
            scored.append((item_id, float(sims.max(axis=0).mean())))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[: max(0, int(limit))]

    def get_items(self, ids: Sequence[int]) -> Dict[int, MemoryItem]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._lock:
            rows = self._require().execute(
                f"SELECT * FROM memory_items WHERE id IN ({marks})", tuple(ids)).fetchall()
        return {int(r["id"]): MemoryItem.from_row(r) for r in rows}

    def recent(self, conversation_id: str, kinds: Optional[Sequence[str]] = None,
               limit: int = 20) -> List[MemoryItem]:
        sql = ("SELECT * FROM memory_items WHERE conversation_id = ? AND invalid_at IS NULL")
        args: List[Any] = [conversation_id]
        if kinds:
            sql += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            args.extend(kinds)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._require().execute(sql, tuple(args)).fetchall()
        return [MemoryItem.from_row(r) for r in rows]

    def count(self, conversation_id: Optional[str] = None) -> int:
        with self._lock:
            conn = self._require()
            if conversation_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM memory_items").fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM memory_items WHERE conversation_id = ?",
                                   (conversation_id,)).fetchone()
        return int(row["n"] if row else 0)

    def forget_session_cache(self, conversation_id: str) -> None:
        """Drop the in-RAM matrices for one session (rows stay on disk)."""
        with self._lock:
            for space in (SPACE_TEXT, SPACE_IMAGE):
                self._idx.pop((conversation_id, space), None)
            self._li.pop(conversation_id, None)
            self._li_loaded.discard(conversation_id)

    @staticmethod
    def _owns_item(conn: sqlite3.Connection, conversation_id: str, item_id: int) -> bool:
        return conn.execute("SELECT 1 FROM memory_items WHERE id = ? AND conversation_id = ?",
                            (item_id, conversation_id)).fetchone() is not None

    def delete_session(self, conversation_id: str) -> int:
        """Delete runtime memory atomically; history and shared media are separate."""
        with self._cleanup_lock:
            with self._lock:
                conn = self._require()
                self._register_session(conn, conversation_id)
                conn.execute("UPDATE memory_sessions SET state='pending' WHERE conversation_id=?", (conversation_id,))
                conn.commit()
            try:
                with self._lock:
                    with conn:
                        for table in ("memory_vectors", "memory_item_keys"):
                            conn.execute(f"DELETE FROM {table} WHERE item_id IN "
                                         "(SELECT id FROM memory_items WHERE conversation_id = ?)", (conversation_id,))
                        count = conn.execute("DELETE FROM memory_items WHERE conversation_id = ?", (conversation_id,)).rowcount
                    self.forget_session_cache(conversation_id)
                if self.frames is not None:
                    self.frames.delete_session(conversation_id)
                with self._lock:
                    conn.execute("DELETE FROM memory_sessions WHERE conversation_id=?", (conversation_id,))
                    conn.commit()
                return count
            except Exception as exc:
                with self._lock:
                    conn.execute("UPDATE memory_sessions SET attempts=attempts+1,last_error=? WHERE conversation_id=?",
                                 (str(exc)[:500], conversation_id))
                    conn.commit()
                raise

    def _register_session(self, conn, conversation_id):
        conn.execute("INSERT OR IGNORE INTO memory_sessions(conversation_id,owner_id,state,created_at) VALUES(?,?,'active',?)",
                     (conversation_id, self.owner_id, time.time()))

    def register_session(self, conversation_id):
        with self._lock:
            conn = self._require()
            self._register_session(conn, conversation_id)
            conn.commit()

    def cleanup_candidates(self, *, recover=False, include_legacy=False):
        with self._lock:
            conn = self._require()
            rows = conn.execute("SELECT conversation_id FROM memory_sessions WHERE state='pending'" +
                                (" OR owner_id != ?" if recover else ""), (self.owner_id,) if recover else ()).fetchall()
            ids = {r[0] for r in rows}
            if include_legacy:
                ids.update(r[0] for r in conn.execute("SELECT DISTINCT conversation_id FROM memory_items "
                           "WHERE conversation_id NOT IN (SELECT conversation_id FROM memory_sessions)"))
            return sorted(ids)

    def refresh_storage_status(self):
        with self._lock:
            self._require()
        # A shared filesystem stat can stall. Never hold the retrieval DB lock
        # while waiting for filesystem capacity metadata.
        free_bytes = shutil.disk_usage(os.path.dirname(os.path.abspath(self.path))).free
        with self._lock:
            conn = self._require()
            page_size = conn.execute('PRAGMA page_size').fetchone()[0]
            pages = conn.execute('PRAGMA page_count').fetchone()[0]
            free_pages = conn.execute('PRAGMA freelist_count').fetchone()[0]
            used = (pages - free_pages) * page_size
            reason = 'low_disk' if free_bytes < self.settings.memory_min_free_bytes else (
                'memory_budget' if self.settings.memory_max_db_bytes > 0 and used >= self.settings.memory_max_db_bytes else '')
            if bool(reason) == self.accepting_writes:
                log.warning('memory storage admission: %s (occupied=%d, disk_free=%d)', reason or 'resumed', used, free_bytes)
            self.accepting_writes = not reason
            self._storage_status = {'occupied_bytes': used, 'reusable_bytes': free_pages * page_size,
                                    'disk_free_bytes': free_bytes, 'paused_reason': reason}
            return dict(self._storage_status)

    def reclaim_space(self, *, vacuum=False):
        """Checkpoint online; full VACUUM is reserved for exclusive offline maintenance."""
        with self._lock:
            conn = self._require()
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)' if vacuum else 'PRAGMA wal_checkpoint(PASSIVE)')
            if vacuum:
                conn.execute('VACUUM')
                conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
