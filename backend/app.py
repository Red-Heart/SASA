# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from core.advisory_engine import AdvisoryEngine
from core.services import get_environmental_factors


app = FastAPI(
    title="Smart Agriculture Advisory System",
    description="ANN + Fuzzy Logic based agriculture advisory",
    version="1.0"
)

engine = AdvisoryEngine()


# Request schema
class AdvisoryRequest(BaseModel):
    location: str
    crop: Optional[str] = None


# API
@app.post("/advisory")
def advisory(req: AdvisoryRequest):
    # 1. Fetch simulated environmental data
    factors = get_environmental_factors(req.location)

    # 2. Run advisory engine
    advisory_result = engine.generate_advisory(
        features=factors,
        crop=req.crop
    )

    # 3. Return transparent response
    return {
        "user_input": {
            "location": req.location,
            "crop": req.crop
        },
        "environmental_factors": factors,
        "advisory": advisory_result
    }