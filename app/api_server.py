from fastapi import FastAPI, HTTPException, Request
from io_model import GenerateRequest, GenerateResponse
from engine import Engine
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    prompt = req.prompt.strip()
    engine: Engine = request.app.state.engine
    
    if not prompt:
        raise HTTPException(status_code=400, detail="invalid prompt")

    if not engine or engine is None:
        raise HTTPException(status_code=500, details="engine does not exist")

    try:
        output = engine.generate_output(req)
    except Exception as e:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail=f"generation failed: {e} ")
    
    # todo - how to make this an async generator?
    return {
        "text": output
    }
