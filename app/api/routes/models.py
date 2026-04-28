from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.model_config import ModelConfig
from app.schemas.model_config import ModelCreate, ModelResponse

router = APIRouter(prefix="/models", tags=["Models"])

@router.post("/", response_model=ModelResponse)
def create_model(model: ModelCreate, db: Session = Depends(get_db)):
    db_model = ModelConfig(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

@router.put("/{model_id}", response_model=ModelResponse)
def update_model(
    model_id: int,
    model: ModelCreate,
    db: Session = Depends(get_db)
):
    db_model = db.query(ModelConfig).filter(ModelConfig.id == model_id).first()

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Update fields
    for key, value in model.dict().items():
        setattr(db_model, key, value)

    db.commit()
    db.refresh(db_model)

    return db_model

@router.get("/", response_model=list[ModelResponse])
def get_models(db: Session = Depends(get_db)):
    return db.query(ModelConfig).all()

@router.delete("/{id}")
def delete_model(id: int, db: Session = Depends(get_db)):
    model = db.query(ModelConfig).filter(ModelConfig.id == id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
        
    # 3. Remove from DB
    db.delete(model)
    db.commit()
    return {"message": "deleted"}