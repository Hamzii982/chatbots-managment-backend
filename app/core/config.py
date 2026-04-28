from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import secrets
import os

class Settings(BaseSettings):
    # Database configuration
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/ai_platform",
        description="PostgreSQL database URL"
    )
    
    # Security configuration
    SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv(
            "SECRET_KEY",
            secrets.token_urlsafe(32)  # Generate strong key if not provided
        ),
        description="Secret key for JWT signing - minimum 32 chars"
    )
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    
    # API configuration
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origins"
    )
    
    # File upload configuration
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, ge=1, le=500, description="Max file size in MB")
    ALLOWED_FILE_EXTENSIONS: list[str] = Field(
        default=[".pdf"],
        description="Allowed file extensions for upload"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra env vars
    )

settings = Settings()