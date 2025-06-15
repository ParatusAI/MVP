# predictor_api.py - FastAPI endpoint for real-time predictions
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
from model_predictor import initialize_predictor, predictor
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CsPbBr3 Spectral Image Predictor API",
    description="Real-time prediction of CsPbBr3 properties from spectral images",
    version="1.0.0"
)

# Initialize predictor on startup
@app.on_event("startup")
async def startup_event():
    try:
        initialize_predictor()
        logger.info("✅ Predictor API started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")

@app.get("/")
async def root():
    """API information"""
    return {
        "service": "CsPbBr3 Spectral Image Predictor",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "predict": "/predict/ (POST with image file)",
            "health": "/health/",
            "docs": "/docs"
        },
        "for": "Ryan's RL Decision Agent"
    }

@app.get("/health/")
async def health_check():
    """Health check for monitoring"""
    try:
        if predictor is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "Predictor not initialized"}
            )
        
        health = predictor.health_check()
        return health
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@app.post("/predict/")
async def predict_from_spectral_image(file: UploadFile = File(...)):
    """
    MAIN ENDPOINT for Ryan's RL Agent
    
    Receives PNG spectral image from Aroyston's spectrometer
    Returns JSON predictions for Ryan's RL decision making
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.png'):
            raise HTTPException(
                status_code=400, 
                detail="Only PNG files are supported"
            )
        
        # Read image data
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty image file"
            )
        
        # Get prediction
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Predictor not initialized"
            )
        
        result = predictor.predict_from_image_data(image_data, file.filename)
        
        logger.info(f"🔮 Processed {file.filename}: PLQY={result.get('predicted_plqy', 0):.3f}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch/")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple spectral images"""
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 files per batch"
            )
        
        results = []
        
        for file in files:
            if not file.filename.lower().endswith('.png'):
                continue
                
            image_data = await file.read()
            if len(image_data) > 0:
                result = predictor.predict_from_image_data(image_data, file.filename)
                results.append(result)
        
        return {
            "batch_size": len(results),
            "predictions": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info/")
async def model_info():
    """Information about the loaded model"""
    try:
        if predictor is None:
            return {"error": "Predictor not initialized"}
        
        return {
            "model_type": "ImprovedSpectralCNN",
            "architecture": "4-layer CNN + 3-layer regressor",
            "input_format": "RGB spectral images (224x224)",
            "output_properties": ["PLQY", "Emission Peak (nm)", "FWHM (nm)"],
            "confidence": predictor.confidence,
            "normalization_ranges": predictor.normalization_ranges,
            "device": str(predictor.device),
            "ready_for_rl": True
        }
        
    except Exception as e:
        return {"error": str(e)}

# For testing with curl
@app.get("/test_prediction/")
async def test_prediction():
    """Test endpoint with dummy data"""
    try:
        from PIL import Image
        import io
        
        # Create dummy spectral image
        dummy_image = Image.new('RGB', (224, 224), color=(255, 100, 50))
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        dummy_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Get prediction
        result = predictor.predict_from_image_data(img_bytes.getvalue(), "test_image.png")
        
        return {
            "test_status": "success",
            "test_prediction": result,
            "note": "This is a dummy prediction for testing"
        }
        
    except Exception as e:
        return {"test_status": "failed", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting CsPbBr3 Predictor API for Ryan's RL Agent...")
    print("📡 Will receive PNG files from Aroyston's spectrometer")
    print("🔮 Will send JSON predictions to Ryan's RL")
    
    uvicorn.run(
        "predictor_api:app",
        host="0.0.0.0",
        port=8001,  # Different port from main API
        reload=False,
        log_level="info"
    )