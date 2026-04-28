from fastapi import APIRouter, status
from app.db.session import SessionLocal
from sqlalchemy import text
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Health check endpoint to verify API is running and database is accessible.
    """
    try:
        # Test database connection
        db = SessionLocal()
        db.execute(text('SELECT * from user LIMIT 1'))
        db.close()
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "disconnected"
        }, status.HTTP_503_SERVICE_UNAVAILABLE


@router.get("/ready", status_code=status.HTTP_200_OK)
def readiness_check():
    """
    Readiness check endpoint for Kubernetes/orchestration platforms.
    """
    return {
        "status": "ready"
    }


@router.get("/config/public", status_code=status.HTTP_200_OK)
def get_public_config():
    """
    Get non-sensitive configuration details.
    Does NOT return secrets or API keys.
    """
    return {
        "cors_origins": settings.ALLOWED_ORIGINS,
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        "allowed_file_extensions": list(settings.ALLOWED_FILE_EXTENSIONS)
    }
