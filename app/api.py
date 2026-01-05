from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from PIL import Image
import io
import json
import base64
from typing import Optional, List
from rapid_detector import RapidDetector
import os

app = FastAPI(title="Rapid Detector API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount stored images directory
from pathlib import Path
# Get the images directory from detector storage
detector_storage_dir = Path.home() / ".cache" / "rapid_detector" / "images"
if detector_storage_dir.exists():
    app.mount("/images", StaticFiles(directory=str(detector_storage_dir)), name="images")

# Serve the main page at root
from fastapi.responses import FileResponse

@app.get("/")
async def read_index():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(static_dir, "index.html"))

# Global detector instance (shared foundation model)
detector = None

def create_id_mask(masks, image_size, downsample_factor=4):
    """Convert binary masks to efficient ID mask.
    
    Args:
        masks: numpy array of shape (N, H, W) with binary masks
        image_size: (width, height) of original image
        downsample_factor: factor to reduce mask size (for performance)
    
    Returns:
        2D numpy array where pixel values indicate detection ID (0=background, 1=first detection, etc.)
    """
    import numpy as np
    from skimage.transform import resize
    
    if len(masks) == 0:
        return None
    
    # Original mask dimensions
    n_masks, orig_h, orig_w = masks.shape
    
    # Downsample for performance (optional)
    target_h = orig_h // downsample_factor
    target_w = orig_w // downsample_factor
    
    # Create empty ID mask
    id_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    
    # Fill mask with detection IDs (later detections overwrite earlier ones)
    for i, mask in enumerate(masks):
        # Downsample mask
        if downsample_factor > 1:
            mask_resized = resize(mask, (target_h, target_w), anti_aliasing=False, preserve_range=True)
            mask_binary = mask_resized > 0.5
        else:
            mask_binary = mask > 0.5
        
        # Set pixels to detection ID (i+1, since 0 is background)
        id_mask[mask_binary] = i + 1
    
    return id_mask

def encode_id_mask_as_png(id_mask):
    """Encode ID mask as PNG base64 data URL for efficient transmission."""
    if id_mask is None:
        return None
    
    import numpy as np
    
    # Convert to PIL Image (using 'L' mode for grayscale)
    pil_image = Image.fromarray(id_mask.astype(np.uint8), mode='L')
    
    # Save as PNG to memory
    png_buffer = io.BytesIO()
    pil_image.save(png_buffer, format='PNG', optimize=True)
    png_bytes = png_buffer.getvalue()
    
    # Encode as base64 data URL
    png_b64 = base64.b64encode(png_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{png_b64}"
    
    return data_url

@app.on_event("startup")
async def startup_event():
    """Initialize the detector."""
    global detector
    detector = RapidDetector()

def load_and_validate_image(image_data: bytes) -> Image.Image:
    """Load and validate uploaded image."""
    try:
        pil_image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return pil_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/detect/{detector_name}")
async def detect_objects(
    detector_name: str,
    image: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(0.5)
):
    """Run object detection using the specified detector configuration."""
    
    # Validate detector exists
    if not detector.config_exists(detector_name):
        raise HTTPException(status_code=404, detail=f"Detector {detector_name} not found")
    
    # Validate confidence threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0.0 and 1.0")
    
    # Load image
    image_data = await image.read()
    pil_image = load_and_validate_image(image_data)
    
    try:
        # Run detection
        start_time = time.time()
        
        result = detector.run_inference(detector_name, pil_image, confidence_threshold)
        
        inference_time = time.time() - start_time
        
        
        # Convert binary masks to efficient ID mask and encode as PNG
        id_mask_png = None
        if len(result["masks"]) > 0:
            id_mask = create_id_mask(result["masks"], pil_image.size)
            if id_mask is not None:
                id_mask_png = encode_id_mask_as_png(id_mask)
        
        # Convert numpy arrays to lists for JSON serialization
        response_data = {
            "boxes": result["boxes"].tolist(),
            "scores": result["scores"].tolist(),
            "inference_time": round(inference_time, 2),
            "num_detections": len(result["boxes"]),
            "id_mask_png": id_mask_png,  # PNG data URL instead of raw array
            "image_size": list(pil_image.size)  # [width, height] for frontend scaling
        }
        
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/detect/{detector_name}/info")
async def get_detector_info(detector_name: str):
    """Get metadata about a detector configuration."""
    if not detector.config_exists(detector_name):
        raise HTTPException(status_code=404, detail=f"Detector {detector_name} not found")
    
    # Load config if not in memory
    if detector_name not in detector.configs:
        detector.load_config(detector_name)
    
    config = detector.configs[detector_name]
    
    info = {
        "name": detector_name,
        "is_semantic_name": config.get("is_semantic_name", True),
        "num_examples": len(config["prompts"]),
        "version": config["version"],
        "saved": config["saved"]
    }
    
    return JSONResponse(content=info)

@app.get("/detect/{detector_name}/examples")
async def get_detector_examples(detector_name: str):
    """Get visual examples (object cutouts) from a detector configuration."""
    if not detector.config_exists(detector_name):
        raise HTTPException(status_code=404, detail=f"Detector {detector_name} not found")
    
    # Load config if not in memory
    if detector_name not in detector.configs:
        detector.load_config(detector_name)
    
    config = detector.configs[detector_name]
    examples = []
    
    for image_id, prompt_data in config["prompts"].items():
        # Get the image
        try:
            image = detector.storage.get_image(image_id)
            if image is None:
                continue
                
            # Extract object cutouts for each bounding box
            for i, (box, is_positive) in enumerate(zip(prompt_data["boxes"], prompt_data["labels"])):
                x1, y1, x2, y2 = box
                
                # Crop the object from the image
                cutout = image.crop((x1, y1, x2, y2))
                
                # Convert to base64 data URL
                buffer = io.BytesIO()
                cutout.save(buffer, format='PNG')
                cutout_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                cutout_data_url = f"data:image/png;base64,{cutout_b64}"
                
                examples.append({
                    "image_id": image_id,
                    "box_index": i,
                    "box": box,
                    "is_positive": is_positive,
                    "cutout": cutout_data_url,
                    "size": [x2 - x1, y2 - y1]
                })
                
        except Exception as e:
            continue
    
    return JSONResponse(content={"examples": examples})

@app.get("/detectors")
async def list_detectors():
    """List all detector configurations."""
    config_names = detector.list_configs()
    return JSONResponse(content={"detectors": config_names})

@app.post("/add_example")
async def add_example(
    image: UploadFile = File(...),
    class_name: str = Form(...),
    annotations: str = Form(...),
    is_positive: bool = Form(True)
):
    """Add a visual example to a detector configuration."""
    # Load image
    image_data = await image.read()
    pil_image = load_and_validate_image(image_data)
    
    # Parse annotations JSON
    try:
        annotation_list = json.loads(annotations)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid annotations JSON")
    
    # Create or get existing config
    detector_name = class_name
    if not detector.config_exists(detector_name):
        detector.new_config(detector_name, is_semantic_name=True)
    
    # Add visual prompts
    boxes = [[ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']] for ann in annotation_list]
    labels = [is_positive] * len(boxes)
    
    detector.update_prompts(detector_name, pil_image, boxes, labels)
    
    # Get total examples count
    config = detector.configs[detector_name]
    total_examples = len(config['prompts'])
    
    return JSONResponse(content={
        "message": "Example added successfully",
        "total_examples": total_examples,
        "detector_name": detector_name
    })

class CreateDetectorRequest(BaseModel):
    name: str
    class_name: str
    is_semantic: bool = True

@app.post("/create_detector")
async def create_detector(request: CreateDetectorRequest):
    """Create a new detector configuration."""
    try:
        detector.new_config(request.name, is_semantic_name=request.is_semantic)
        return JSONResponse(content={
            "message": "Detector created successfully",
            "detector_name": request.name
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save_detector/{detector_name}")
async def save_detector(detector_name: str, uploaded_images: Optional[str] = Form(None)):
    """Save a detector configuration to disk."""
    if not detector.config_exists(detector_name):
        raise HTTPException(status_code=404, detail=f"Detector {detector_name} not found")
    
    # Save uploaded images if provided
    if uploaded_images:
        try:
            images_data = json.loads(uploaded_images)
            config = detector.configs[detector_name]
            config['uploaded_images'] = images_data
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid uploaded_images JSON")
    
    # Save to disk
    detector.save_config(detector_name)
    
    # Get config info
    config = detector.configs[detector_name]
    total_examples = len(config.get('prompts', {}))
    
    return JSONResponse(content={
        "message": "Detector saved successfully",
        "detector_name": detector_name,
        "total_examples": total_examples
    })

@app.post("/save_uploaded_images/{detector_name}")
async def save_uploaded_images(
    detector_name: str,
    images: List[UploadFile] = File(...)
):
    """Save uploaded images for a detector configuration."""
    if not detector.config_exists(detector_name):
        # Create new detector if it doesn't exist
        detector.new_config(detector_name, is_semantic_name=True)
    
    saved_images = []
    
    for image in images:
        try:
            # Load and validate image
            image_data = await image.read()
            pil_image = load_and_validate_image(image_data)
            
            # Store image and get hash ID
            image_id = detector.storage.add_image(pil_image)
            
            saved_images.append({
                "id": image_id,
                "filename": image.filename,
                "size": list(pil_image.size)
            })
            
        except Exception as e:
            continue
    
    # Save image list to detector config
    config = detector.configs[detector_name]
    config['uploaded_images'] = saved_images
    
    return JSONResponse(content={
        "message": f"Saved {len(saved_images)} images",
        "images": saved_images
    })

@app.get("/uploaded_images/{detector_name}")
async def get_uploaded_images(detector_name: str):
    """Get uploaded images for a detector configuration."""
    if not detector.config_exists(detector_name):
        raise HTTPException(status_code=404, detail=f"Detector {detector_name} not found")
    
    # Load config if not in memory
    if detector_name not in detector.configs:
        detector.load_config(detector_name)
    
    config = detector.configs[detector_name]
    uploaded_images = config.get('uploaded_images', [])
    
    # Return image URLs pointing to static file serving
    image_data = []
    for img_info in uploaded_images:
        try:
            # Verify image file exists
            image_path = detector.storage.images_dir / f"{img_info['id']}.png"
            assert image_path.exists()
            image_data.append({
                "id": img_info['id'],
                "filename": img_info.get('filename', f"image_{img_info['id']}.png"),
                "size": img_info.get('size', [0, 0]),
                "url": f"/images/{img_info['id']}.png"  # Static file URL
            })

        except Exception as e:
            continue
    
    return JSONResponse(content={"images": image_data})

@app.delete("/detect/{detector_name}")
async def delete_detector(detector_name: str):
    """Delete a detector configuration."""
    if not detector.config_exists(detector_name):
        raise HTTPException(status_code=404, detail=f"Detector {detector_name} not found")
    
    # Delete from memory and disk
    detector.delete_config(detector_name)
    
    return JSONResponse(content={
        "message": "Detector deleted successfully",
        "detector_name": detector_name
    })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "model_loaded": detector.model is not None if detector else False,
        "timestamp": time.time()
    })