from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.user import User # Import your new User model
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dependency to get DB session for the init script
def init_admin(logger=None):
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            logger.info("Creating default admin user...")
            hashed_pwd = pwd_context.hash("password") # Initial password: password
            new_admin = User(
                username="admin",
                full_name="Administrator",
                email="smuth@zimatec.de",
                hashed_password=hashed_pwd,
                is_active=True
            )
            db.add(new_admin)
            db.commit()
            logger.info("Admin user created successfully.")
    except Exception as e:
        logger.error(f"Error initializing admin: {e}")
    finally:
        db.close()