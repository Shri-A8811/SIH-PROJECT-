"""
Persistent State Store for Sovereign On-Premise Agentic AI Workbench.
Adheres strictly to the architectural rule:
"Models are stateless workers. A persistent orchestrator and state store maintain continuity."
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import sqlite3
from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from config.settings import settings

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
    inputs_json = Column(Text, default="{}")
    context_json = Column(Text, default="{}")
    assigned_model = Column(String(64), nullable=False)
    allowed_tools_json = Column(Text, default="[]")
    output_schema_json = Column(Text, default="{}")
    retry_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    result_json = Column(Text, nullable=True)
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
    structured_data_json = Column(Text, default="{}")
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=get_utc_now)

    project = relationship("Project", back_populates="evidence")


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
    details_json = Column(Text, default="{}")
    timestamp = Column(DateTime, default=get_utc_now)


from sqlalchemy.pool import StaticPool, NullPool


class StateStore:
    """Thread-safe persistent state store interface using SQLAlchemy."""

    def __init__(self, database_url: Optional[str] = None):
        self.db_url = database_url or settings.database_url
        if self.db_url.startswith("sqlite"):
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
        else:
            self.engine = create_engine(self.db_url, echo=False)
            
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

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
                    inputs_json=json.dumps(inputs or {}),
                    context_json=json.dumps(context or {}),
                    allowed_tools_json=json.dumps(allowed_tools or []),
                    output_schema_json=json.dumps(output_schema or {}),
                    retry_count=0,
                    status=TaskStatus.PENDING,
                )
                session.add(task)
            else:
                task.task_type = task_type
                task.objective = objective
                task.assigned_model = assigned_model
                task.inputs_json = json.dumps(inputs or {})
                task.context_json = json.dumps(context or {})
                task.allowed_tools_json = json.dumps(allowed_tools or [])
                task.output_schema_json = json.dumps(output_schema or {})
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
                task.result_json = json.dumps(result)
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
                    structured_data_json=json.dumps(structured_data or {}),
                    confidence=confidence,
                )
                session.add(evidence)
            else:
                evidence.source_type = source_type
                evidence.source_document = source_document
                evidence.page_number = page_number
                evidence.section = section
                evidence.extracted_text = extracted_text
                evidence.structured_data_json = json.dumps(structured_data or {})
                evidence.confidence = confidence
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
                details_json=json.dumps(details or {}),
            )
            session.add(log_entry)
            session.commit()

    def get_recent_model_activity(self, limit: int = 50) -> List[ModelActivityLog]:
        with self.Session() as session:
            stmt = select(ModelActivityLog).order_by(ModelActivityLog.timestamp.desc()).limit(limit)
            return list(session.scalars(stmt).all())
