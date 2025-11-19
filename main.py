import os
import time
import uuid
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from bson import ObjectId

from database import db, create_document, get_documents


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        try:
            return ObjectId(str(v))
        except Exception:
            raise ValueError("Invalid ObjectId")


class GenerationSettings(BaseModel):
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "custom"] = "1:1"
    width: Optional[int] = None
    height: Optional[int] = None
    style: Literal["Realistic", "Artistic", "Anime", "3D Render", "Oil Painting", "Cyberpunk", "Sketch", "Watercolor"] = "Realistic"
    num_images: int = Field(1, ge=1, le=4)
    quality: Literal["Draft", "Standard", "HD", "Ultra HD"] = "Standard"
    seed: Optional[int] = None
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)
    steps: int = Field(30, ge=5, le=150)
    model: str = "Stable-Diffusion-XL"


class ImageInfo(BaseModel):
    url: str
    format: Literal["png", "jpg", "webp"] = "png"
    width: int
    height: int
    upscale_available: bool = True
    variation_available: bool = True


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    settings: GenerationSettings


class GenerationDocument(BaseModel):
    id: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    settings: GenerationSettings
    status: Literal["queued", "processing", "completed", "failed"] = "completed"
    progress: int = 100
    eta_seconds: int = 0
    images: List[ImageInfo] = []
    favorite: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


app = FastAPI(title="AI Image Generation Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "AI Image Generation API"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


# Simple demo credits stored in a single-user doc
DEMO_USER_ID = "demo-user"

def _get_or_create_demo_user() -> Dict[str, Any]:
    user = db["user"].find_one({"_id": DEMO_USER_ID}) if db else None
    if not user and db is not None:
        user = {"_id": DEMO_USER_ID, "name": "Demo User", "credits": 1000, "created_at": datetime.now(timezone.utc)}
        db["user"].insert_one(user)
    return user or {"_id": DEMO_USER_ID, "name": "Demo User", "credits": 1000}


def _ar_to_wh(ar: str, custom_w: Optional[int], custom_h: Optional[int]) -> (int, int):
    mapping = {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1024, 768),
        "3:4": (768, 1024),
    }
    if ar == "custom" and custom_w and custom_h:
        return custom_w, custom_h
    return mapping.get(ar, (1024, 1024))


def _placeholder_image(seed: str, w: int, h: int) -> str:
    # Use picsum with seed for deterministic placeholder
    return f"https://picsum.photos/seed/{seed}/{w}/{h}"


@app.get("/api/credits")
def get_credits():
    user = _get_or_create_demo_user()
    return {"user_id": user["_id"], "credits": user.get("credits", 0)}


@app.post("/api/generate", response_model=GenerationDocument)
def generate_images(req: GenerationRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Database not available")

    user = _get_or_create_demo_user()
    credits_needed = max(1, req.settings.num_images)
    if user.get("credits", 0) < credits_needed:
        raise HTTPException(status_code=402, detail="Not enough credits")

    w, h = _ar_to_wh(req.settings.aspect_ratio, req.settings.width, req.settings.height)

    # Simulate different formats for downloads
    formats = ["png", "jpg", "webp"]

    images: List[ImageInfo] = []
    batch_seed = str(uuid.uuid4())[:8]
    for i in range(req.settings.num_images):
        seed = f"{batch_seed}-{i}-{req.settings.seed or i}"
        url = _placeholder_image(seed, w, h)
        images.append(ImageInfo(url=url, width=w, height=h, format=formats[i % len(formats)]))

    doc = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "settings": req.settings.model_dump(),
        "status": "completed",
        "progress": 100,
        "eta_seconds": 0,
        "images": [img.model_dump() for img in images],
        "favorite": False,
        "created_at": datetime.now(timezone.utc),
    }

    inserted_id = db["generation"].insert_one(doc).inserted_id

    # Deduct credits
    db["user"].update_one({"_id": DEMO_USER_ID}, {"$inc": {"credits": -credits_needed}})

    doc_out = doc.copy()
    doc_out["id"] = str(inserted_id)
    return doc_out


@app.get("/api/history")
def history(q: Optional[str] = None, favorite: Optional[bool] = None, style: Optional[str] = None,
           sort: Literal["newest", "oldest"] = "newest", limit: int = 50):
    if not db:
        raise HTTPException(status_code=500, detail="Database not available")

    filter_dict: Dict[str, Any] = {}
    if q:
        filter_dict["prompt"] = {"$regex": q, "$options": "i"}
    if favorite is not None:
        filter_dict["favorite"] = favorite
    if style:
        filter_dict["settings.style"] = style

    cursor = db["generation"].find(filter_dict)
    cursor = cursor.sort("created_at", -1 if sort == "newest" else 1).limit(limit)

    items = []
    for it in cursor:
        it["id"] = str(it.pop("_id"))
        items.append(it)

    return {"items": items, "count": len(items)}


@app.patch("/api/generation/{gen_id}/favorite")
def toggle_favorite(gen_id: str, value: Optional[bool] = None):
    if not db:
        raise HTTPException(status_code=500, detail="Database not available")

    obj_id = ObjectId(gen_id)
    doc = db["generation"].find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    new_val = bool(value) if value is not None else (not doc.get("favorite", False))
    db["generation"].update_one({"_id": obj_id}, {"$set": {"favorite": new_val, "updated_at": datetime.now(timezone.utc)}})
    return {"id": gen_id, "favorite": new_val}


@app.delete("/api/generation/{gen_id}")
def delete_generation(gen_id: str):
    if not db:
        raise HTTPException(status_code=500, detail="Database not available")

    obj_id = ObjectId(gen_id)
    res = db["generation"].delete_one({"_id": obj_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"success": True, "id": gen_id}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
