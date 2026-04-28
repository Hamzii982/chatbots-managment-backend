from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.routes import models, chatbots, system, statistics, usage, auth
from app.db.base import Base
from app.db.session import engine
from app.models import chatbot, model_config, message, chatbot_documents, user
from app.core.config import settings
from app.core.admin import init_admin
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This creates the tables if they don't exist
Base.metadata.create_all(bind=engine)

init_admin(logger)

app = FastAPI(
    title="AI Control Panel",
    description="RAG-based Chatbot Management System",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

app.include_router(models.router)
app.include_router(chatbots.router)
app.include_router(system.router)
app.include_router(statistics.router)
app.include_router(usage.router)
app.include_router(auth.router)

# Configure CORS with specific allowed origins (not wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Explicit methods only
    allow_headers=["*"],  # Explicit headers only
    max_age=600, 
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        raise


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors with detailed feedback.
    """
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )


@app.get("/")
def root():
    """Root endpoint - redirects to health check."""
    return {"status": "ok", "message": "API is running. Check /system/health for status."}