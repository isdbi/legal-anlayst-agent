from fastapi import FastAPI, File, HTTPException, UploadFile
import fitz
from pydantic import BaseModel
import uvicorn
import markdown2

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





def extract_text(file: UploadFile) -> str:
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        # PDF to text
        with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    elif filename.endswith(".md"):
        md_content = file.file.read().decode("utf-8")
        return markdown2.markdown(md_content)
    elif filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type. Only PDF, Markdown (.md), or TXT are allowed.")

@app.post("/analyze/file", response_model=ContractAnalysis)
async def analyze_file(file: UploadFile = File(...), jurisdiction: str = "UAE", check_shariah: bool = True):
    try:
        contract_text = extract_text(file)
        result = agent.analyze_full_contract(
            contract_text=contract_text,
            jurisdiction=jurisdiction,
            check_shariah=check_shariah
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
