"""SQLite-based analytics logger for the RAG chat API."""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "analytics.db"


class AnalyticsLogger:
    """Thread-safe analytics logger backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DB_PATH)
        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._create_table()

    # ------------------------------------------------------------------
    # Thread-local connection
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_table(self) -> None:
        self._get_conn().execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                session_id      TEXT,
                question        TEXT,
                answer_preview  TEXT,
                papers_cited    TEXT,
                chunks_retrieved INTEGER,
                response_length INTEGER,
                latency_ms      INTEGER,
                query_type      TEXT,
                error           TEXT,
                ip_hash         TEXT
            )
            """
        )
        self._get_conn().commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def log_chat(
        self,
        *,
        session_id: str = "",
        question: str = "",
        answer_preview: str = "",
        papers_cited: list[str] | None = None,
        chunks_retrieved: int = 0,
        response_length: int = 0,
        latency_ms: int = 0,
        query_type: str = "",
        error: str = "",
        ip_hash: str = "",
    ) -> None:
        """Insert a single chat event."""
        try:
            with self._lock:
                self._get_conn().execute(
                    """
                    INSERT INTO chat_logs
                        (timestamp, session_id, question, answer_preview,
                         papers_cited, chunks_retrieved, response_length,
                         latency_ms, query_type, error, ip_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        session_id,
                        question,
                        answer_preview,
                        json.dumps(papers_cited or []),
                        chunks_retrieved,
                        response_length,
                        latency_ms,
                        query_type,
                        error,
                        ip_hash,
                    ),
                )
                self._get_conn().commit()
        except Exception as exc:
            logger.warning("Failed to log chat event: %s", exc)

    # ------------------------------------------------------------------
    # Read — summary
    # ------------------------------------------------------------------
    def get_summary(self, days: int = 30) -> dict:
        """Return aggregate statistics for the last *days* days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._lock:
            cur = self._get_conn().execute(
                "SELECT COUNT(*) AS cnt FROM chat_logs WHERE timestamp >= ?",
                (cutoff,),
            )
            total_questions = cur.fetchone()["cnt"]

            cur = self._get_conn().execute(
                "SELECT COUNT(DISTINCT session_id) AS cnt FROM chat_logs WHERE timestamp >= ?",
                (cutoff,),
            )
            unique_sessions = cur.fetchone()["cnt"]

            cur = self._get_conn().execute(
                "SELECT AVG(latency_ms) AS avg_lat FROM chat_logs WHERE timestamp >= ? AND latency_ms > 0",
                (cutoff,),
            )
            row = cur.fetchone()
            avg_latency_ms = round(row["avg_lat"]) if row["avg_lat"] is not None else 0

            cur = self._get_conn().execute(
                "SELECT COUNT(*) AS cnt FROM chat_logs WHERE timestamp >= ? AND error != ''",
                (cutoff,),
            )
            error_count = cur.fetchone()["cnt"]
            error_rate = round(error_count / max(total_questions, 1), 4)

            # Top papers cited
            cur = self._get_conn().execute(
                "SELECT papers_cited FROM chat_logs WHERE timestamp >= ? AND papers_cited != '[]'",
                (cutoff,),
            )
            paper_counter: dict[str, int] = {}
            for r in cur.fetchall():
                try:
                    for p in json.loads(r["papers_cited"]):
                        paper_counter[p] = paper_counter.get(p, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
            top_papers_cited = sorted(
                [{"paper": p, "citations": c} for p, c in paper_counter.items()],
                key=lambda x: x["citations"],
                reverse=True,
            )[:10]

            # Questions per day
            cur = self._get_conn().execute(
                """
                SELECT DATE(timestamp) AS day, COUNT(*) AS cnt
                FROM chat_logs
                WHERE timestamp >= ?
                GROUP BY day
                ORDER BY day
                """,
                (cutoff,),
            )
            questions_per_day = [{"date": r["day"], "count": r["cnt"]} for r in cur.fetchall()]

            # Recent questions
            cur = self._get_conn().execute(
                """
                SELECT question, timestamp, session_id, latency_ms, error
                FROM chat_logs
                ORDER BY id DESC
                LIMIT 20
                """,
            )
            recent_questions = [dict(r) for r in cur.fetchall()]

        return {
            "total_questions": total_questions,
            "unique_sessions": unique_sessions,
            "avg_latency_ms": avg_latency_ms,
            "error_rate": error_rate,
            "top_papers_cited": top_papers_cited,
            "questions_per_day": questions_per_day,
            "recent_questions": recent_questions,
        }

    # ------------------------------------------------------------------
    # Read — paginated questions
    # ------------------------------------------------------------------
    def get_questions(self, page: int = 1, per_page: int = 20) -> dict:
        """Return paginated question log."""
        offset = (max(page, 1) - 1) * per_page

        with self._lock:
            cur = self._get_conn().execute("SELECT COUNT(*) AS cnt FROM chat_logs")
            total = cur.fetchone()["cnt"]

            cur = self._get_conn().execute(
                """
                SELECT * FROM chat_logs
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (per_page, offset),
            )
            items = [dict(r) for r in cur.fetchall()]

        return {
            "page": page,
            "per_page": per_page,
            "total": total,
            "items": items,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_csv(self) -> str:
        """Return all logs as a CSV string."""
        with self._lock:
            cur = self._get_conn().execute("SELECT * FROM chat_logs ORDER BY id")
            rows = cur.fetchall()
        if not rows:
            return ""

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(tuple(row))
        return output.getvalue()
