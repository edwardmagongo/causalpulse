"""Tables (including users and API keys), the engine, and the constants both sides share.

The worker imports this module and nothing else from `cp_api`
(`.importlinter`'s `worker-sees-only-the-db` contract). So the queue name and
the job's dotted path live here: both sides need them, and the API enqueues
by string so it never imports the worker.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

QUEUE_NAME = "analyses"
JOB_FUNCTION = "cp_worker.jobs.run_analysis_job"
# BSTS and hierarchical DiD run MCMC; RQ's 180-second default would kill them.
JOB_TIMEOUT_SECONDS = 30 * 60

STATUSES = ("queued", "running", "succeeded", "failed")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    """An identity that owns datasets and analyses. `name` is a handle, not a login."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ApiKey(Base):
    """One issued key. Only the secret's SHA-256 is stored (see `cp_api.auth`)."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    secret_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("owner_id", "sha256"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    # Unique per owner (see __table_args__): two users uploading the same bytes
    # each get their own dataset.
    sha256: Mapped[str] = mapped_column(String(64))
    n_bytes: Mapped[int] = mapped_column(Integer)
    csv: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Always equal to its dataset's owner; the API enforces that at creation.
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"))
    # The column is `schema`, the spec's name; the attribute is not, so it
    # never reads as SQLAlchemy's own `schema` concept.
    causal_schema: Mapped[dict[str, Any]] = mapped_column("schema", JSONB)
    requested_estimators: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    random_seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    partial: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RoutingDecisionRow(Base):
    __tablename__ = "routing_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"))
    estimator_name: Mapped[str] = mapped_column(String(64))
    eligible: Mapped[bool] = mapped_column(Boolean)
    reasons: Mapped[list[str]] = mapped_column(ARRAY(Text))


class EstimateRow(Base):
    """One requested estimator's result: a summary and reports, or an error."""

    __tablename__ = "estimates"
    __table_args__ = (UniqueConstraint("analysis_id", "method"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"))
    method: Mapped[str] = mapped_column(String(64))
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    band: Mapped[str | None] = mapped_column(String(16))
    report_html: Mapped[str | None] = mapped_column(Text)
    report_markdown: Mapped[str | None] = mapped_column(Text)
    report_assets: Mapped[dict[str, str] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)


class AnalysisSummary(Base):
    __tablename__ = "analysis_summaries"

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analyses.id"), primary_key=True
    )
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB)
    comparisons: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or database_url(), pool_pre_ping=True)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
