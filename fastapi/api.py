# Sample API code to showcase basic FastAPI functionality
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import time

app = FastAPI(
    title="Machine Learning API",
    description="An API to register ML models, make predictions, and upload data. Additional endpoints demonstrate Pydantic models, dependency injection, and async support.",
    version="1.0.0"
)

# ------------------------------
# Pydantic Model Definitions
# ------------------------------

# Machine Learning Model for CRUD endpoints
class MLModel(BaseModel):
    id: int
    name: str
    version: str = "1.0"
    description: Optional[str] = None
    trained: bool = False

# Request body example with a basic Item model.
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# Advanced validation example for a User model.
class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    # Use 'pattern' with a raw string (r"") instead of 'regex'
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    age: int = Field(gt=0, lt=150)
    is_active: bool = Field(default=True)

# ------------------------------
# In-Memory "Database" & Dependency
# ------------------------------

# In-memory store for ML models.
model_db: List[MLModel] = []

async def get_model_db() -> List[MLModel]:
    """Dependency that provides the current model database."""
    return model_db

# Dependency function to extract common query parameters.
def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

# ------------------------------
# Endpoints: ML Models CRUD
# ------------------------------

@app.get("/", summary="API Root", tags=["Root"])
async def root():
    """
    Welcome endpoint for the ML API.
    """
    return {"message": "Welcome to the Machine Learning API"}

# List all registered ML models.
@app.get("/models", response_model=List[MLModel], summary="List all ML models", tags=["Models"])
async def list_models(db: List[MLModel] = Depends(get_model_db)):
    return db

# Register a new ML model.
@app.post("/models", response_model=MLModel, status_code=201, summary="Register a new ML model", tags=["Models"])
async def register_model(new_model: MLModel, db: List[MLModel] = Depends(get_model_db)):
    # Check if a model with the same ID already exists.
    if any(model.id == new_model.id for model in db):
        raise HTTPException(status_code=400, detail="Model with this ID already exists")
    db.append(new_model)
    return new_model

# Get details about a specific ML model by ID.
@app.get("/models/{model_id}", response_model=MLModel, summary="Get ML model details", tags=["Models"])
async def get_model(model_id: int, db: List[MLModel] = Depends(get_model_db)):
    for model in db:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail="Model not found")

# Update an existing ML model.
@app.put("/models/{model_id}", response_model=MLModel, summary="Update an ML model", tags=["Models"])
async def update_model(model_id: int, updated_model: MLModel, db: List[MLModel] = Depends(get_model_db)):
    for index, model in enumerate(db):
        if model.id == model_id:
            db[index] = updated_model
            return updated_model
    raise HTTPException(status_code=404, detail="Model not found")

# Delete an ML model.
@app.delete("/models/{model_id}", summary="Delete an ML model", tags=["Models"])
async def delete_model(model_id: int, db: List[MLModel] = Depends(get_model_db)):
    for index, model in enumerate(db):
        if model.id == model_id:
            db.pop(index)
            return {"message": "Model deleted successfully"}
    raise HTTPException(status_code=404, detail="Model not found")

# ------------------------------
# Endpoints: Additional Examples
# ------------------------------

# Dummy prediction endpoint.
@app.get("/predictions", summary="Make a prediction", tags=["Predictions"])
async def get_prediction(task_id: int, input_feature: Optional[float] = None):
    """
    Dummy prediction endpoint.
    - **task_id**: A unique ID for the prediction task.
    - **input_feature**: A numerical input feature; if provided, the prediction is input_feature * 2.
    """
    
    if input_feature is not None:
        prediction = input_feature * 2
        return {"task_id": task_id, "input_feature": input_feature, "prediction": prediction}
    return {"task_id": task_id, "message": "No input feature provided"}

# Endpoint to upload training data as raw bytes.
@app.post("/upload_training_data", summary="Upload training data", tags=["File Uploads"])
async def upload_training_data(file: bytes = File(...)):
    """
    Upload training data as raw bytes.
    In a real application, you might process or store the data.
    """
    file_size = len(file)
    return {"message": "Training data uploaded successfully", "file_size": file_size}

# Endpoint to upload a pre-trained model file.
@app.post("/upload_model_file", summary="Upload a pre-trained model file", tags=["File Uploads"])
async def upload_model_file(file: UploadFile = File(...)):
    """
    Upload a file (e.g., a pre-trained model binary).
    Returns the filename and content type.
    """
    return {"filename": file.filename, "content_type": file.content_type}

# Endpoint demonstrating a request body using the Item model.
@app.post("/items/", summary="Create an item", tags=["Items"])
async def create_item(item: Item):
    """
    Create an item with pricing details.
    If tax is provided, calculate the total cost.
    """
    item_dict = item.dict()
    if item.tax:
        total = item.price + item.tax
        item_dict.update({"total": total})
    return item_dict

# Endpoint demonstrating advanced validation using the User model.
@app.post("/users/", summary="Create a user", tags=["Users"])
async def create_user(user: User):
    """
    Create a user with advanced field validation.
    """
    return user

# Endpoint demonstrating dependency injection with common query parameters.
@app.get("/items/commons", summary="Read items with common parameters", tags=["Items"])
async def read_items(commons: dict = Depends(common_parameters)):
    """
    Returns common query parameters (q, skip, limit) passed via the dependency.
    """
    return commons

# ------------------------------
# Endpoints: Async and Sync Examples
# ------------------------------

@app.get("/async-example", summary="Async example", tags=["Examples"])
async def async_endpoint():
    """
    Example async endpoint that waits for 1 second.
    """
    await asyncio.sleep(1)
    return {"message": "This is an async endpoint"}

@app.get("/sync-example", summary="Sync example", tags=["Examples"])
def sync_endpoint():
    """
    Example sync endpoint that blocks for 1 second.
    """
    time.sleep(1)  # Blocking operation
    return {"message": "This is a sync endpoint"}