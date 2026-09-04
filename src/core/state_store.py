"""
Persistent State Store for Sovereign On-Premise Agentic AI Workbench.
Upgraded to PostgreSQL + pgvector with SQLAlchemy for enterprise-grade vector search
and ACID state tracking across projects, tasks, evidence, knowledge chunks, and telemetry.
Adheres strictly to the architectural rule:
"Models are stateless workers. A persistent orchestrator and state store maintain continuity."
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import numpy as np
from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Index,
    create_engine,
    select,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector
from config.settings import settings, BASE_DIR

logger = logging.getLogger(__name__)

Base = declarative_base()


def get_utc_now() -> datetime:
    """Returns timezone-naive UTC timestamp for clean DB compatibility."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class Project(Base):
    __tablename__ = "projects"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    objective = Column(Text, nullable=False)
    status = Column(String(32), default="active")
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    tasks = relationship("TaskRecord", back_populates="project", cascade="all, delete-orphan")
    evidence = relationship("EvidenceRecord", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("ArtifactRecord", back_populates="project", cascade="all, delete-orphan")


class TaskRecord(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), nullable=False)
    project_id = Column(String(64), ForeignKey("projects.id"), nullable=False)
    task_type = Column(String(64), nullable=False)
    objective = Column(Text, nullable=False)
    inputs_json = Column(JSON, default=dict)
    context_json = Column(JSON, default=dict)
    assigned_model = Column(String(64), nullable=False)
    allowed_tools_json = Column(JSON, default=list)
    output_schema_json = Column(JSON, default=dict)
    retry_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    result_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="tasks")


class EvidenceRecord(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(String(64), nullable=False)
    project_id = Column(String(64), ForeignKey("projects.id"), nullable=False)
    source_type = Column(String(64), nullable=False)  # ocr, vision, retrieval, calculation, sandbox
    source_document = Column(String(255), nullable=True)
    page_number = Column(Integer, nullable=True)
    section = Column(String(255), nullable=True)
    extracted_text = Column(Text, nullable=False)
    structured_data_json = Column(JSON, default=dict)
    confidence = Column(Float, default=1.0)
    embedding = Column(Vector(settings.vector_dimension), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    project = relationship("Project", back_populates="evidence")


class KnowledgeChunkRecord(Base):
    """pgvector knowledge table for high-dimensional semantic search and hybrid RAG."""
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(64), unique=True, index=True, nullable=False)
    document_name = Column(String(255), nullable=False)
    category = Column(String(128), default="General", index=True, nullable=False)
    section_title = Column(String(255), nullable=True)
    page_number = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.vector_dimension), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

Index(
    "ix_knowledge_chunks_embedding",
    KnowledgeChunkRecord.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


class DocumentRecord(Base):
    """Tracks uploaded knowledge base documents, folder categories, and chunk stats."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), unique=True, index=True, nullable=False)
    category = Column(String(128), default="General", index=True, nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=get_utc_now)


class ChatSessionRecord(Base):
    """Persists chat sessions for multi-chat history."""
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True)
    title = Column(String(255), nullable=False)
    knowledge_scope = Column(String(128), default="All Documents")
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    messages = relationship("ChatMessageRecord", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessageRecord.created_at.asc()")


class ChatMessageRecord(Base):
    """Individual message in a multi-chat thread."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(32), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=get_utc_now)

    session = relationship("ChatSessionRecord", back_populates="messages")



class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    artifact_id = Column(String(64), nullable=False)
    project_id = Column(String(64), ForeignKey("projects.id"), nullable=False)
    artifact_type = Column(String(64), nullable=False)  # docx, xlsx, pptx, code
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    is_verified = Column(Integer, default=0)  # 0: unverified, 1: passed, -1: failed
    verification_notes = Column(Text, default="")
    created_at = Column(DateTime, default=get_utc_now)

    project = relationship("Project", back_populates="artifacts")


class ModelActivityLog(Base):
    __tablename__ = "model_activity_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), nullable=True)
    task_id = Column(String(64), nullable=True)
    model_name = Column(String(64), nullable=False)
    action = Column(String(32), nullable=False)  # LOAD, UNLOAD, INFERENCE
    vram_allocated_mb = Column(Float, default=0.0)
    duration_ms = Column(Float, default=0.0)
    details_json = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=get_utc_now)


class StateStore:
    """Enterprise persistent state store interface with PostgreSQL + pgvector support."""

    def __init__(self, database_url: Optional[str] = None):
        self.db_url = database_url or settings.database_url
        self.is_postgres = bool(self.db_url and ("postgresql" in self.db_url or "postgres" in self.db_url))

        if self.is_postgres:
            try:
                self.engine = create_engine(
                    self.db_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    echo=False,
                )
                # Verify connection & initialize pgvector extension
                with self.engine.connect() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    conn.commit()
                self._ensure_tables_and_columns()
            except Exception as e:
                logger.warning(
                    f"PostgreSQL connection at {self.db_url} failed ({e}). "
                    "Switching to resilient local state store fallback."
                )
                # Fallback to local SQLite engine if PostgreSQL daemon is offline
                fallback_path = BASE_DIR / "workbench_state.db"
                self.db_url = f"sqlite:///{fallback_path}"
                self.is_postgres = False
                self.engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False},
                    echo=False,
                )
                self._ensure_tables_and_columns()
        elif self.db_url.startswith("sqlite"):
            if ":memory:" in self.db_url:
                self.engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                    echo=False,
                )
            else:
                self.engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False},
                    echo=False,
                )
            self._ensure_tables_and_columns()
        else:
            self.engine = create_engine(self.db_url, echo=False)
            self._ensure_tables_and_columns()

        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def _ensure_tables_and_columns(self):
        """Creates tables and applies self-healing column migrations for SQLite fallback."""
        Base.metadata.create_all(self.engine)
        if not self.is_postgres and self.db_url.startswith("sqlite") and ":memory:" not in self.db_url:
            try:
                with self.engine.connect() as conn:
                    cols_kc = [row[1] for row in conn.execute(text("PRAGMA table_info(knowledge_chunks);")).fetchall()]
                    if cols_kc and "category" not in cols_kc:
                        conn.execute(text("ALTER TABLE knowledge_chunks ADD COLUMN category VARCHAR(128) DEFAULT 'General';"))
                        conn.commit()
                    cols_ev = [row[1] for row in conn.execute(text("PRAGMA table_info(evidence);")).fetchall()]
                    if cols_ev and "embedding" not in cols_ev:
                        conn.execute(text("ALTER TABLE evidence ADD COLUMN embedding JSON;"))
                        conn.commit()
            except Exception as e:
                logger.debug(f"SQLite migration notice: {e}")

    # ----------------------------------------------------------------------------------------------
    # PROJECT MANAGEMENT
    # ----------------------------------------------------------------------------------------------
    def create_project(self, project_id: str, name: str, objective: str) -> Project:
        with self.Session() as session:
            project = session.get(Project, project_id)
            if not project:
                project = Project(id=project_id, name=name, objective=objective)
                session.add(project)
            else:
                project.name = name
                project.objective = objective
                project.updated_at = get_utc_now()
            session.commit()
            session.refresh(project)
            return project

    def get_project(self, project_id: str) -> Optional[Project]:
        with self.Session() as session:
            return session.get(Project, project_id)

    # ----------------------------------------------------------------------------------------------
    # TASK LIFECYCLE & CONTRACT MANAGEMENT
    # ----------------------------------------------------------------------------------------------
    def add_task(
        self,
        task_id: str,
        project_id: str,
        task_type: str,
        objective: str,
        assigned_model: str,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[List[str]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        with self.Session() as session:
            stmt = (
                select(TaskRecord)
                .where(TaskRecord.task_id == task_id, TaskRecord.project_id == project_id)
                .order_by(TaskRecord.id.desc())
            )
            task = session.scalars(stmt).first()
            if not task:
                task = TaskRecord(
                    task_id=task_id,
                    project_id=project_id,
                    task_type=task_type,
                    objective=objective,
                    assigned_model=assigned_model,
                    inputs_json=inputs or {},
                    context_json=context or {},
                    allowed_tools_json=allowed_tools or [],
                    output_schema_json=output_schema or {},
                    retry_count=0,
                    status=TaskStatus.PENDING,
                )
                session.add(task)
            else:
                task.task_type = task_type
                task.objective = objective
                task.assigned_model = assigned_model
                task.inputs_json = inputs or {}
                task.context_json = context or {}
                task.allowed_tools_json = allowed_tools or []
                task.output_schema_json = output_schema or {}
                task.status = TaskStatus.PENDING
            session.commit()
            session.refresh(task)
            return task

    def mark_task_running(self, task_id: str, project_id: Optional[str] = None):
        """Strict status discipline: Mark RUNNING before any model/tool side effect begins."""
        with self.Session() as session:
            stmt = select(TaskRecord).where(TaskRecord.task_id == task_id)
            if project_id:
                stmt = stmt.where(TaskRecord.project_id == project_id)
            stmt = stmt.order_by(TaskRecord.id.desc())
            task = session.scalars(stmt).first()
            if task:
                task.status = TaskStatus.RUNNING
                task.started_at = get_utc_now()
                session.commit()

    def complete_task(self, task_id: str, result: Dict[str, Any], project_id: Optional[str] = None):
        """Atomic write: update result and mark COMPLETED in one transaction."""
        with self.Session() as session:
            stmt = select(TaskRecord).where(TaskRecord.task_id == task_id)
            if project_id:
                stmt = stmt.where(TaskRecord.project_id == project_id)
            stmt = stmt.order_by(TaskRecord.id.desc())
            task = session.scalars(stmt).first()
            if task:
                task.result_json = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = get_utc_now()
                session.commit()

    def fail_task(self, task_id: str, error_message: str, increment_retry: bool = True, project_id: Optional[str] = None):
        with self.Session() as session:
            stmt = select(TaskRecord).where(TaskRecord.task_id == task_id)
            if project_id:
                stmt = stmt.where(TaskRecord.project_id == project_id)
            stmt = stmt.order_by(TaskRecord.id.desc())
            task = session.scalars(stmt).first()
            if task:
                if increment_retry:
                    task.retry_count = (task.retry_count or 0) + 1
                task.last_error = error_message
                task.status = TaskStatus.FAILED
                session.commit()

    def escalate_task(self, task_id: str, reason: str, project_id: Optional[str] = None):
        with self.Session() as session:
            stmt = select(TaskRecord).where(TaskRecord.task_id == task_id)
            if project_id:
                stmt = stmt.where(TaskRecord.project_id == project_id)
            stmt = stmt.order_by(TaskRecord.id.desc())
            task = session.scalars(stmt).first()
            if task:
                task.last_error = f"ESCALATED: {reason}"
                task.status = TaskStatus.ESCALATED
                session.commit()

    def get_task(self, task_id: str, project_id: Optional[str] = None) -> Optional[TaskRecord]:
        with self.Session() as session:
            stmt = select(TaskRecord).where(TaskRecord.task_id == task_id)
            if project_id:
                stmt = stmt.where(TaskRecord.project_id == project_id)
            stmt = stmt.order_by(TaskRecord.id.desc())
            return session.scalars(stmt).first()

    def list_tasks_for_project(self, project_id: str) -> List[TaskRecord]:
        with self.Session() as session:
            stmt = select(TaskRecord).where(TaskRecord.project_id == project_id).order_by(TaskRecord.id.asc())
            return list(session.scalars(stmt).all())

    def recover_pending_and_running_tasks(self, project_id: str) -> List[TaskRecord]:
        """Restart safety: Reconstruct any interrupted task that was left RUNNING."""
        with self.Session() as session:
            stmt = (
                select(TaskRecord)
                .where(
                    TaskRecord.project_id == project_id,
                    TaskRecord.status.in_([TaskStatus.RUNNING, TaskStatus.PENDING]),
                )
                .order_by(TaskRecord.id.asc())
            )
            interrupted_tasks = list(session.scalars(stmt).all())
            for t in interrupted_tasks:
                t.status = TaskStatus.PENDING
            session.commit()
            return interrupted_tasks

    # ----------------------------------------------------------------------------------------------
    # EVIDENCE MANAGEMENT
    # ----------------------------------------------------------------------------------------------
    def add_evidence(
        self,
        evidence_id: str,
        project_id: str,
        source_type: str,
        extracted_text: str,
        source_document: Optional[str] = None,
        page_number: Optional[int] = None,
        section: Optional[str] = None,
        structured_data: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        embedding: Optional[List[float]] = None,
    ) -> EvidenceRecord:
        with self.Session() as session:
            stmt = (
                select(EvidenceRecord)
                .where(EvidenceRecord.evidence_id == evidence_id, EvidenceRecord.project_id == project_id)
                .order_by(EvidenceRecord.id.desc())
            )
            evidence = session.scalars(stmt).first()
            if not evidence:
                evidence = EvidenceRecord(
                    evidence_id=evidence_id,
                    project_id=project_id,
                    source_type=source_type,
                    source_document=source_document,
                    page_number=page_number,
                    section=section,
                    extracted_text=extracted_text,
                    structured_data_json=structured_data or {},
                    confidence=confidence,
                    embedding=embedding,
                )
                session.add(evidence)
            else:
                evidence.source_type = source_type
                evidence.source_document = source_document
                evidence.page_number = page_number
                evidence.section = section
                evidence.extracted_text = extracted_text
                evidence.structured_data_json = structured_data or {}
                evidence.confidence = confidence
                if embedding is not None:
                    evidence.embedding = embedding
            session.commit()
            session.refresh(evidence)
            return evidence

    def get_evidence(self, evidence_id: str, project_id: Optional[str] = None) -> Optional[EvidenceRecord]:
        with self.Session() as session:
            stmt = select(EvidenceRecord).where(EvidenceRecord.evidence_id == evidence_id)
            if project_id:
                stmt = stmt.where(EvidenceRecord.project_id == project_id)
            stmt = stmt.order_by(EvidenceRecord.id.desc())
            return session.scalars(stmt).first()

    def get_all_evidence_for_project(self, project_id: str) -> List[EvidenceRecord]:
        with self.Session() as session:
            stmt = select(EvidenceRecord).where(EvidenceRecord.project_id == project_id).order_by(EvidenceRecord.id.asc())
            return list(session.scalars(stmt).all())

    # ----------------------------------------------------------------------------------------------
    # PGVECTOR KNOWLEDGE CHUNK MANAGEMENT & VECTOR SIMILARITY SEARCH
    # ----------------------------------------------------------------------------------------------
    def upsert_knowledge_chunk(
        self,
        chunk_id: str,
        document_name: str,
        content: str,
        section_title: Optional[str] = None,
        page_number: Optional[int] = None,
        embedding: Optional[List[float]] = None,
        category: str = "General",
    ) -> KnowledgeChunkRecord:
        """Stores or updates a knowledge chunk with pgvector embedding and category."""
        if self.is_postgres:
            # Native PostgreSQL Upsert (ON CONFLICT DO UPDATE)
            stmt = insert(KnowledgeChunkRecord).values(
                chunk_id=chunk_id,
                document_name=document_name,
                category=category,
                section_title=section_title,
                page_number=page_number,
                content=content,
                embedding=embedding,
            )
            do_update_stmt = stmt.on_conflict_do_update(
                index_elements=['chunk_id'],
                set_=dict(
                    document_name=stmt.excluded.document_name,
                    category=stmt.excluded.category,
                    section_title=stmt.excluded.section_title,
                    page_number=stmt.excluded.page_number,
                    content=stmt.excluded.content,
                    embedding=stmt.excluded.embedding,
                )
            )
            with self.Session() as session:
                session.execute(do_update_stmt)
                session.commit()
                # Fetch to return the ORM object
                return session.scalars(select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.chunk_id == chunk_id)).first()
        else:
            # Fallback for SQLite
            with self.Session() as session:
                stmt = select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.chunk_id == chunk_id)
                record = session.scalars(stmt).first()
                if not record:
                    record = KnowledgeChunkRecord(
                        chunk_id=chunk_id,
                        document_name=document_name,
                        category=category,
                        section_title=section_title,
                        page_number=page_number,
                        content=content,
                        embedding=embedding,
                    )
                    session.add(record)
                else:
                    record.document_name = document_name
                    record.category = category
                    record.section_title = section_title
                    record.page_number = page_number
                    record.content = content
                    if embedding is not None:
                        record.embedding = embedding
                session.commit()
                session.refresh(record)
                return record

    def search_vector_chunks(
        self,
        query_vector: List[float],
        top_k: int = 50,
        category: Optional[str] = None,
    ) -> List[Tuple[KnowledgeChunkRecord, float]]:
        """
        Executes vector similarity search using pgvector cosine distance when on PostgreSQL,
        or high-performance numpy cosine similarity when running on fallback engine.
        Filters by category if category is specified and not 'All Documents'.
        Returns list of (KnowledgeChunkRecord, similarity_score).
        """
        filter_cat = category if category and category not in ("All Documents", "All", "") else None

        if self.is_postgres:
            try:
                with self.Session() as session:
                    stmt = (
                        select(
                            KnowledgeChunkRecord,
                            KnowledgeChunkRecord.embedding.cosine_distance(query_vector).label("distance"),
                        )
                        .where(KnowledgeChunkRecord.embedding.is_not(None))
                    )
                    if filter_cat:
                        stmt = stmt.where(KnowledgeChunkRecord.category == filter_cat)
                    stmt = stmt.order_by("distance").limit(top_k)
                    rows = session.execute(stmt).all()
                    # Cosine similarity = 1.0 - cosine_distance
                    return [(row[0], max(0.0, 1.0 - float(row[1]))) for row in rows]
            except Exception as e:
                logger.warning(f"PostgreSQL pgvector search exception ({e}). Using Python vector scoring.")

        # Fallback scoring for SQLite / test environments
        with self.Session() as session:
            stmt = select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.embedding.is_not(None))
            if filter_cat:
                stmt = stmt.where(KnowledgeChunkRecord.category == filter_cat)
            chunks = list(session.scalars(stmt).all())
            if not chunks:
                return []

            q = np.array(query_vector, dtype=np.float32)
            norm_q = np.linalg.norm(q)
            if norm_q < 1e-6:
                return [(c, 0.0) for c in chunks[:top_k]]

            scored = []
            for c in chunks:
                if c.embedding is not None:
                    v = np.array(c.embedding, dtype=np.float32)
                    norm_v = np.linalg.norm(v)
                    sim = float(np.dot(q, v) / (norm_q * norm_v)) if norm_v > 1e-6 else 0.0
                    scored.append((c, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def get_all_knowledge_chunks(self) -> List[KnowledgeChunkRecord]:
        with self.Session() as session:
            stmt = select(KnowledgeChunkRecord).order_by(KnowledgeChunkRecord.id.asc())
            return list(session.scalars(stmt).all())

    def get_document_chunks_by_filename(self, filename: str) -> List[KnowledgeChunkRecord]:
        """Retrieves all indexed semantic chunks belonging to a specific document filename."""
        with self.Session() as session:
            stmt = select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.document_name == filename).order_by(KnowledgeChunkRecord.page_number.asc(), KnowledgeChunkRecord.id.asc())
            return list(session.scalars(stmt).all())

    def clear_knowledge_chunks(self):
        with self.Session() as session:
            session.execute(text("DELETE FROM knowledge_chunks;"))
            session.commit()

    # ----------------------------------------------------------------------------------------------
    # ARTIFACT AUDITING
    # ----------------------------------------------------------------------------------------------
    def record_artifact(
        self,
        artifact_id: str,
        project_id: str,
        artifact_type: str,
        file_path: str,
        file_size_bytes: int = 0,
        is_verified: int = 0,
        verification_notes: str = "",
    ) -> ArtifactRecord:
        with self.Session() as session:
            stmt = (
                select(ArtifactRecord)
                .where(ArtifactRecord.artifact_id == artifact_id, ArtifactRecord.project_id == project_id)
                .order_by(ArtifactRecord.id.desc())
            )
            artifact = session.scalars(stmt).first()
            if not artifact:
                artifact = ArtifactRecord(
                    artifact_id=artifact_id,
                    project_id=project_id,
                    artifact_type=artifact_type,
                    file_path=file_path,
                    file_size_bytes=file_size_bytes,
                    is_verified=is_verified,
                    verification_notes=verification_notes,
                )
                session.add(artifact)
            else:
                artifact.artifact_type = artifact_type
                artifact.file_path = file_path
                artifact.file_size_bytes = file_size_bytes
                artifact.is_verified = is_verified
                artifact.verification_notes = verification_notes
            session.commit()
            session.refresh(artifact)
            return artifact

    def update_artifact_verification(
        self,
        artifact_id: str,
        is_verified: int,
        verification_notes: str = "",
        project_id: Optional[str] = None,
    ):
        with self.Session() as session:
            stmt = select(ArtifactRecord).where(ArtifactRecord.artifact_id == artifact_id)
            if project_id:
                stmt = stmt.where(ArtifactRecord.project_id == project_id)
            stmt = stmt.order_by(ArtifactRecord.id.desc())
            artifact = session.scalars(stmt).first()
            if artifact:
                artifact.is_verified = is_verified
                artifact.verification_notes = verification_notes
                session.commit()

    # ----------------------------------------------------------------------------------------------
    # MODEL ACTIVITY TELEMETRY
    # ----------------------------------------------------------------------------------------------
    def log_model_activity(
        self,
        model_name: str,
        action: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        vram_allocated_mb: float = 0.0,
        duration_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        with self.Session() as session:
            log_entry = ModelActivityLog(
                project_id=project_id,
                task_id=task_id,
                model_name=model_name,
                action=action,
                vram_allocated_mb=vram_allocated_mb,
                duration_ms=duration_ms,
                details_json=details or {},
            )
            session.add(log_entry)
            session.commit()

    def get_recent_model_activity(self, limit: int = 50) -> List[ModelActivityLog]:
        with self.Session() as session:
            stmt = select(ModelActivityLog).order_by(ModelActivityLog.timestamp.desc()).limit(limit)
            return list(session.scalars(stmt).all())

    # ----------------------------------------------------------------------------------------------
    # DOCUMENT INVENTORY & CATEGORIZATION MANAGEMENT
    # ----------------------------------------------------------------------------------------------
    def upsert_document(
        self,
        filename: str,
        category: str = "General",
        file_path: str = "",
        file_size_bytes: int = 0,
        chunk_count: int = 0,
    ) -> DocumentRecord:
        with self.Session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.filename == filename)
            doc = session.scalars(stmt).first()
            if not doc:
                doc = DocumentRecord(
                    filename=filename,
                    category=category,
                    file_path=file_path,
                    file_size_bytes=file_size_bytes,
                    chunk_count=chunk_count,
                )
                session.add(doc)
            else:
                doc.category = category
                doc.file_path = file_path
                doc.file_size_bytes = file_size_bytes
                doc.chunk_count = chunk_count
                doc.uploaded_at = get_utc_now()
            session.commit()
            session.refresh(doc)
            return doc

    def list_documents(self, category: Optional[str] = None) -> List[DocumentRecord]:
        with self.Session() as session:
            stmt = select(DocumentRecord)
            if category and category not in ("All Documents", "All", ""):
                stmt = stmt.where(DocumentRecord.category == category)
            stmt = stmt.order_by(DocumentRecord.uploaded_at.desc())
            return list(session.scalars(stmt).all())

    def get_document(self, filename: str) -> Optional[DocumentRecord]:
        with self.Session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.filename == filename)
            return session.scalars(stmt).first()

    def delete_document(self, filename: str) -> bool:
        """Deletes document record and cleans up associated knowledge chunks from DB."""
        with self.Session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.filename == filename)
            doc = session.scalars(stmt).first()
            if doc:
                session.delete(doc)
            # Delete associated knowledge chunks
            chunk_stmt = select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.document_name == filename)
            chunks = list(session.scalars(chunk_stmt).all())
            for c in chunks:
                session.delete(c)
            session.commit()
            return True

    def get_categories(self) -> List[str]:
        """Returns distinct list of document categories currently registered."""
        with self.Session() as session:
            stmt = select(DocumentRecord.category).distinct()
            cats = [c for c in session.scalars(stmt).all() if c]
            if "General" not in cats:
                cats.insert(0, "General")
            return sorted(list(set(cats)))

    # ----------------------------------------------------------------------------------------------
    # MULTI-CHAT SESSION & CONVERSATION HISTORY MANAGEMENT
    # ----------------------------------------------------------------------------------------------
    def create_chat_session(
        self,
        session_id: Optional[str] = None,
        title: str = "New Conversation",
        knowledge_scope: str = "All Documents",
    ) -> ChatSessionRecord:
        sid = session_id or f"CHAT_{int(datetime.now(timezone.utc).timestamp())}_{int(np.random.randint(1000, 9999))}"
        with self.Session() as session:
            record = session.get(ChatSessionRecord, sid)
            if not record:
                record = ChatSessionRecord(
                    id=sid,
                    title=title,
                    knowledge_scope=knowledge_scope,
                )
                session.add(record)
                session.commit()
                session.refresh(record)
            return record

    def get_chat_sessions(self) -> List[ChatSessionRecord]:
        with self.Session() as session:
            stmt = select(ChatSessionRecord).order_by(ChatSessionRecord.updated_at.desc())
            return list(session.scalars(stmt).all())

    def get_chat_session(self, session_id: str) -> Optional[ChatSessionRecord]:
        with self.Session() as session:
            return session.get(ChatSessionRecord, session_id)

    def update_chat_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        knowledge_scope: Optional[str] = None,
    ) -> Optional[ChatSessionRecord]:
        with self.Session() as session:
            rec = session.get(ChatSessionRecord, session_id)
            if rec:
                if title is not None:
                    rec.title = title
                if knowledge_scope is not None:
                    rec.knowledge_scope = knowledge_scope
                rec.updated_at = get_utc_now()
                session.commit()
                session.refresh(rec)
            return rec

    def delete_chat_session(self, session_id: str) -> bool:
        with self.Session() as session:
            rec = session.get(ChatSessionRecord, session_id)
            if rec:
                session.delete(rec)
                session.commit()
                return True
            return False

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessageRecord:
        with self.Session() as session:
            # Ensure parent session exists
            sess = session.get(ChatSessionRecord, session_id)
            if not sess:
                sess = ChatSessionRecord(
                    id=session_id,
                    title=content[:40] if role == "user" else "New Conversation",
                )
                session.add(sess)
            else:
                sess.updated_at = get_utc_now()
                # Auto-generate title from first user message if still default
                if role == "user" and sess.title in ("New Conversation", "Turnaround Inspection Chat"):
                    sess.title = content[:45] + ("..." if len(content) > 45 else "")

            msg = ChatMessageRecord(
                session_id=session_id,
                role=role,
                content=content,
                metadata_json=metadata or {},
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg

    def get_chat_messages(self, session_id: str) -> List[ChatMessageRecord]:
        with self.Session() as session:
            stmt = (
                select(ChatMessageRecord)
                .where(ChatMessageRecord.session_id == session_id)
                .order_by(ChatMessageRecord.created_at.asc())
            )
            return list(session.scalars(stmt).all())

    def clear_chat_messages(self, session_id: str) -> int:
        """Deletes all messages for a specific chat session."""
        with self.Session() as session:
            stmt = select(ChatMessageRecord).where(ChatMessageRecord.session_id == session_id)
            messages = list(session.scalars(stmt).all())
            count = len(messages)
            for m in messages:
                session.delete(m)
            session.commit()
            return count

