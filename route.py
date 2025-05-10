from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from agenttest import LegalAnalysisAgent
from agenttest import ContractAnalysis  # Assuming ContractAnalysis is exported from agent.py

# Define request schema
class ContractRequest(BaseModel):
    contract_text: str
    jurisdiction: str = "UAE"
    check_shariah: bool = True

# Initialize FastAPI app
app = FastAPI(
    title="Legal Analysis API",
    description="API for analyzing contracts for legal and Shariah compliance",
    version="1.0.0"
)

# Instantiate the agent once at startup
agent = LegalAnalysisAgent()

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/analyze", response_model=ContractAnalysis, tags=["Analysis"])
def analyze_contract(request: ContractRequest):
    """Analyze a contract and return compliance analysis"""
    try:
        result = agent.analyze_full_contract(
            contract_text=request.contract_text,
            jurisdiction=request.jurisdiction,
            check_shariah=request.check_shariah
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
