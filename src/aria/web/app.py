"""Main FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aria.config import get_settings
from aria.logging import get_logger
from aria.memory import ArchiveIndex, VectorStore
from aria.memory.embeddings import OllamaEmbeddings
from aria.web import dependencies
from aria.web.api import admin, auth, documents
from aria.web.database import UserDatabase
from aria.web.models import UserCreate, UserRole

logger = get_logger("aria.web.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()

    # Initialize user database
    user_db_path = settings.aria_data_dir / "cache" / "users.db"
    user_db = UserDatabase(user_db_path)
    await user_db.initialize()
    dependencies.set_user_db(user_db)

    # Create default admin user if no users exist
    if not await user_db.user_exists():
        logger.info("No users found. Creating default admin user...")
        admin_user = UserCreate(
            username="admin",
            password="admin123",  # CHANGE THIS!
            full_name="Administrator",
            role=UserRole.ADMIN,
        )
        await user_db.create_user(admin_user)
        logger.warning(
            "Created default admin user: username='admin', password='admin123' "
            "- PLEASE CHANGE THIS PASSWORD IMMEDIATELY!"
        )

    # Initialize vector store
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        host=settings.ollama_host,
    )
    vector_store = VectorStore(
        persist_directory=settings.aria_data_dir / "chroma",
        embedding_provider=embeddings,
        collection_name="archived_documents",
    )
    await vector_store.initialize()
    dependencies.set_vector_store(vector_store)

    # Initialize archive index
    archive_index = ArchiveIndex(db_path=settings.archive_db_path)
    await archive_index.initialize()
    dependencies.set_archive_index(archive_index)

    logger.info("ARIA web application started")
    logger.info("Access at: http://localhost:8000")
    logger.info("Default login: username=admin, password=admin123")

    yield

    # Cleanup
    await user_db.close()
    logger.info("ARIA web application stopped")


# Create FastAPI app
app = FastAPI(
    title="ARIA Archive",
    description="Document archive with semantic search",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(admin.router)

# Setup templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# HTML routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to login page."""
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    """Documents page."""
    return templates.TemplateResponse("documents.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Admin panel page."""
    return templates.TemplateResponse("admin.html", {"request": request})
