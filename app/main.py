import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Deterministic sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown tasks."""
    # Import here to avoid circular imports
    from .routers.provider_status_ws import status_manager
    
    # Auto-create new tables (hallucination_settings, knowledge_base_entries)
    from .db import engine
    from .models import Base, HallucinationSettings, KnowledgeBaseEntryDB
    from sqlalchemy import inspect as sa_inspect
    async with engine.begin() as conn:
        def _create_missing(sync_conn):
            inspector = sa_inspect(sync_conn)
            existing = set(inspector.get_table_names())
            for table in [HallucinationSettings.__table__, KnowledgeBaseEntryDB.__table__]:
                if table.name not in existing:
                    table.create(sync_conn)
        await conn.run_sync(_create_missing)
    
    # Start the provider status monitor loop
    monitor_task = asyncio.create_task(status_manager.monitor_loop())
    yield
    # Cleanup on shutdown
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass


# Single service entrypoint
app = FastAPI(
    title="Chat_Service Service",
    description="Service for Genesis2026",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chat_service"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": f"Chat_Service Service is running"}

# Service-specific endpoint
@app.get("/api/v1/status")
async def status():
    return {"service": "chat_service", "status": "active", "version": "1.0"}

# Include all routers
from .routers.resonant_chat import router as resonant_chat_router
from .routers.analytics import router as analytics_router
from .routers.streaming import router as streaming_router
from .routers.websocket import router as websocket_router
from .routers.provider_status_ws import router as provider_status_router
from .routers.skills import router as skills_router
from .routers.owner_catalog import router as owner_catalog_router
from .routers.ide_completions import router as ide_completions_router

app.include_router(resonant_chat_router)
app.include_router(analytics_router)
app.include_router(streaming_router)
app.include_router(websocket_router)
app.include_router(provider_status_router)
app.include_router(skills_router)
app.include_router(owner_catalog_router)
app.include_router(ide_completions_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
