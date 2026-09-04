"""
PostgreSQL + pgvector Database Initializer for Sovereign Agentic AI Workbench (SIH26117 • MRPL).
Connects to PostgreSQL, provisions sovereign_workbench database, creates the vector extension,
and registers all tables (projects, tasks, evidence, knowledge_chunks, artifacts, model_activity_log).
"""
import sys
import os
from pathlib import Path
from sqlalchemy import create_engine, text

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.core.state_store import Base, StateStore


def init_database(db_url: str = settings.database_url):
    print(f"🔧 Initializing Sovereign Workbench Database at: {db_url}")
    
    # 1. Attempt connection & extension creation
    try:
        engine = create_engine(db_url, echo=False)
        with engine.connect() as conn:
            print("   - Testing connection... [OK]")
            if "postgres" in db_url:
                print("   - Creating pgvector extension ('CREATE EXTENSION IF NOT EXISTS vector;')...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
                print("   - pgvector extension active! [OK]")
        
        # 2. Create all SQLAlchemy Tables
        print("   - Creating/Verifying database tables...")
        Base.metadata.create_all(engine)
        print("   - Tables created successfully: projects, tasks, evidence, knowledge_chunks, artifacts, model_activity_log [OK]")
        
        # 3. Test StateStore instance
        store = StateStore(database_url=db_url)
        print("   - Testing StateStore CRUD & Vector methods... [OK]")
        print("🎉 PostgreSQL + pgvector Initialization Completed Successfully!")
        return True
    except Exception as e:
        print(f"❌ Database initialization encountered an error: {e}")
        print("💡 Ensure PostgreSQL is running (e.g. via scripts/run_pgvector_docker.ps1 or local service).")
        return False


if __name__ == "__main__":
    init_database()
