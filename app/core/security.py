"""
Security utilities for the application.
"""
import hashlib
import hmac
from typing import Optional


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage (one-way).
    Note: This is a basic implementation. For production, consider using
    a proper secrets management service (AWS Secrets Manager, HashiCorp Vault, etc.)
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verify if a provided API key matches the stored hash.
    """
    return hmac.compare_digest(hash_api_key(provided_key), stored_hash)


def sanitize_error_message(error: Exception) -> str:
    """
    Sanitize error messages to prevent information leakage.
    Returns generic message in production, detailed in development.
    """
    error_msg = str(error)
    # Don't expose internal details in error messages
    if "api_key" in error_msg.lower():
        return "Authentication error"
    if "password" in error_msg.lower():
        return "Authentication error"
    if any(keyword in error_msg.lower() for keyword in ["database", "connection", "postgres"]):
        return "Database connection error"
    return "Internal server error"
