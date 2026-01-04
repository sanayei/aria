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
from aria.web.api import admin, auth, documents
from aria.web.database import UserDatabase
from aria.web.models import UserCreate, UserPermissions, UserRole

logger = get_logger("aria.web.app")

# Global instances
user_db: UserDatabase | None = None
vector_store: VectorStore | None = None
archive_index: ArchiveIndex | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global user_db, vector_store, archive_index

    settings = get_settings()

    # Initialize user database
    user_db_path = settings.aria_data_dir / "cache" / "users.db"
    user_db = UserDatabase(user_db_path)
    await user_db.initialize()

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

    # Initialize archive index
    archive_index = ArchiveIndex(db_path=settings.archive_db_path)
    await archive_index.initialize()

    logger.info("ARIA web application started")

    yield

    # Cleanup
    if user_db:
        await user_db.close()
    logger.info("ARIA web application stopped")


def create_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        FastAPI app instance
    """
    app = FastAPI(
        title="ARIA Archive",
        description="Document archive with semantic search",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Include routers
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

    # Dependency injection for database and stores
    @app.middleware("http")
    async def inject_dependencies(request: Request, call_next):
        """Inject global dependencies into request state."""
        request.state.user_db = user_db
        request.state.vector_store = vector_store
        request.state.archive_index = archive_index
        response = await call_next(request)
        return response

    # Custom dependency functions
    async def get_user_db() -> UserDatabase:
        """Get user database dependency."""
        return user_db

    async def get_vector_store() -> VectorStore:
        """Get vector store dependency."""
        return vector_store

    async def get_archive_index() -> ArchiveIndex:
        """Get archive index dependency."""
        return archive_index

    # Override FastAPI dependencies
    from aria.web.api.auth import router as auth_router

    auth_router.dependencies.append(lambda: user_db)

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

    return app


# Create app instance
app = create_app()


# Dependency overrides for API endpoints
from fastapi import Depends


async def get_user_db_dep():
    """Get user database (dependency)."""
    return user_db


async def get_vector_store_dep():
    """Get vector store (dependency)."""
    return vector_store


async def get_archive_index_dep():
    """Get archive index (dependency)."""
    return archive_index


# Apply dependencies to routers
def setup_dependencies():
    """Setup dependency injection for all routers."""
    from aria.web.api import auth, documents, admin

    # Auth endpoints
    for route in auth.router.routes:
        if hasattr(route, "dependant"):
            route.dependant.dependencies.insert(
                0, Depends(get_user_db_dep, use_cache=False)
            )

    # Documents endpoints
    for route in documents.router.routes:
        if hasattr(route, "dependant"):
            route.dependant.dependencies.extend(
                [
                    Depends(get_vector_store_dep, use_cache=False),
                    Depends(get_archive_index_dep, use_cache=False),
                ]
            )

    # Admin endpoints
    for route in admin.router.routes:
        if hasattr(route, "dependant"):
            route.dependant.dependencies.insert(
                0, Depends(get_user_db_dep, use_cache=False)
            )
