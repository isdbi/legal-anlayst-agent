import json
import os
import glob
import sys
from typing import Dict, List, Any, Optional
import logging
from dotenv import load_dotenv

# LangChain imports
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_extraction_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.agents import AgentExecutor, create_openai_functions_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./legal_knowledge_base")
LEGAL_DOCS_DIR = os.getenv("LEGAL_DOCS_DIR", "./legal_documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Pydantic models
class LegalClause(BaseModel):
    clause_id: str
    clause_type: str
    clause_content: str
    section_number: Optional[str] = None

class ClauseIssue(BaseModel):
    issue_type: str
    description: str
    severity: str
    suggested_fix: str
    legal_basis: str

class ClauseAnalysis(BaseModel):
    clause: LegalClause
    issues: List[ClauseIssue]
    is_compliant: bool
    is_shariah_compliant: Optional[bool]
    recommendations: List[str]

class ContractAnalysis(BaseModel):
    contract_id: str
    missing_clauses: List[str]
    analyzed_clauses: List[ClauseAnalysis]
    overall_compliance_score: float
    jurisdiction_conflicts: List[str]
    shariah_compliance_issues: Optional[List[str]]

class LegalAnalysisAgent:
    def __init__(self):
        logger.info("Initializing Agent...")
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE,
                              streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.knowledge_base = self._load_or_create_vector_db()
        self.tools = self._create_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = self._create_agent_executor()
        logger.info("Agent initialized successfully")

    def _load_or_create_vector_db(self) -> Chroma:
        if os.path.isdir(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=self.embeddings)
        return self._ingest_legal_documents()

    def _ingest_legal_documents(self) -> Chroma:
        if not os.path.isdir(LEGAL_DOCS_DIR):
            raise FileNotFoundError(f"Directory {LEGAL_DOCS_DIR} not found")
        all_docs = []
        loaders = {"pdf": PyPDFLoader, "docx": Docx2txtLoader, "txt": TextLoader}
        for ext, Loader in loaders.items():
            for path in glob.glob(os.path.join(LEGAL_DOCS_DIR, f"**/*.{ext}"), recursive=True):
                docs = Loader(path).load()
                for d in docs:
                    d.metadata.update({"source": path})
                all_docs.extend(docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(all_docs)
        db = Chroma.from_documents(chunks, embedding=self.embeddings, persist_directory=VECTOR_DB_DIR)
        db.persist()
        return db

    def _create_tools(self) -> List[Tool]:
        return [
            Tool("ClauseExtractor", func=self.extract_clauses, description="Extract clauses"),
            Tool("GapDetector", func=self.detect_gaps, description="Detect missing clauses"),
            Tool("ComplianceChecker", func=self.check_compliance, description="Check compliance"),
            Tool("ShariahComplianceChecker", func=self.check_shariah_compliance, description="Shariah compliance"),
            Tool("SuggestionGenerator", func=self.generate_suggestions, description="Suggest fixes"),
            Tool("KnowledgeBaseSearch", func=self.search_knowledge_base, description="Search KB"),
            Tool("FullContractAnalyzer", func=self.analyze_full_contract, description="Analyze full contract")
        ]

    def _create_agent_executor(self) -> AgentExecutor:
        system = "You are a legal analysis agent..."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system), ("human", "{input}"), ("ai", "{agent_scratchpad}")
        ])
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools,
                             memory=self.memory, verbose=True,
                             handle_parsing_errors=True)

    def extract_clauses(self, document_text: str) -> List[LegalClause]:
        schema = {"properties": {"clauses": {"type":"array","items":{"type":"object","properties":{"clause_id":{"type":"string"},"clause_type":{"type":"string"},"clause_content":{"type":"string"},"section_number":{"type":"string"}},"required":["clause_type","clause_content"]}}},"required":["clauses"]}
        chain = create_extraction_chain(schema, self.llm)
        result = chain.invoke({"input": document_text})
        clauses = []
        for idx, data in enumerate(result.get("clauses", [])):
            if "clause_id" not in data:
                data["clause_id"] = f"clause_{idx+1}"
            clauses.append(LegalClause(**data))
        return clauses

    def detect_gaps(self, document_text: str, document_type: str, jurisdiction: str) -> List[str]:
    

     template = (
        "You are a legal expert. For a {document_type} in {jurisdiction}, "
        "analyze this document and return a JSON array of strings listing the **missing or underdeveloped clauses**. "
        "Each item should be one short sentence like: 'Missing confidentiality clause.' or "
        "'Termination clause lacks details on notice period.'\n\n"
        "Document:\n{document_text}"
    )
     prompt = PromptTemplate(template=template,
                            input_variables=["document_text", "document_type", "jurisdiction"])
    
     input_text = prompt.format(
        document_text=document_text,
        document_type=document_type,
        jurisdiction=jurisdiction
    )

     chain = LLMChain(llm=self.llm, prompt=prompt)
     resp = chain.invoke({
    "document_text": document_text,
    "document_type": document_type,
    "jurisdiction": jurisdiction
})
     logger.info(f"Response from LLM: {resp}")
        # Check if the response is a dictionary or a string
        
    
        

    # Extract response text
     if isinstance(resp, dict):
        resp_text = resp.get("output", "") or resp.get("text", "")
     else:
        resp_text = str(resp)

    # Try to parse JSON
     try:
        result = json.loads(resp_text)
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            return result
        else:
            raise ValueError("Invalid structure: not a list of strings")
     except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from LLM output: {e}")
        return [line.strip() for line in resp_text.splitlines() if line.strip()]


     

    def check_compliance(self, clause_text: str, jurisdiction: str) -> Dict[str, Any]:
        retriever = self.knowledge_base.as_retriever(search_kwargs={"k":5, "filter":{"jurisdiction": jurisdiction}})
        template = "Analyze compliance of: {clause_text} wrt {jurisdiction}. Context: {context}"  
        prompt = ChatPromptTemplate.from_template(template)
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        return retrieval_chain.invoke({"clause_text": clause_text, "jurisdiction": jurisdiction})

    def check_shariah_compliance(self, clause_text: str) -> Dict[str, Any]:
        retriever = self.knowledge_base.as_retriever(search_kwargs={"k":5, "filter":{"document_type":"shariah_standard"}})
        template = "Analyze Shariah compliance of: {clause_text}. Context: {context}"  
        prompt = ChatPromptTemplate.from_template(template)
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        return retrieval_chain.invoke({"clause_text": clause_text})

    def generate_suggestions(self, clause_text: str, issue_description: str) -> str:
        template = "Revise clause: {clause_text} to fix: {issue_description}"  
        prompt = PromptTemplate(template=template, input_variables=["clause_text","issue_description"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({"clause_text": clause_text, "issue_description": issue_description})

    def search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        docs = self.knowledge_base.similarity_search(query, k=5)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    def analyze_full_contract(self, contract_text: str, jurisdiction: str, check_shariah: bool=False) -> ContractAnalysis:
        clauses = self.extract_clauses(contract_text)
        missing = self.detect_gaps(contract_text, "contract", jurisdiction)
        analyses = []
        for c in clauses:
            comp = self.check_compliance(c.clause_content, jurisdiction)
            sh = self.check_shariah_compliance(c.clause_content) if check_shariah else None
            issues = []
            is_comp = True
            is_sh = None if not check_shariah else True
            for issue in comp.get("issues", []):
                issues.append(ClauseIssue(issue_type="conflict", description=issue.get("description",""), severity=issue.get("severity","medium"), suggested_fix=issue.get("suggested_fix",""), legal_basis=issue.get("legal_basis","")))
                is_comp = False
            if check_shariah:
                is_sh = True
                for issue in sh.get("issues", []):
                    issues.append(ClauseIssue(issue_type="shariah_conflict", description=issue.get("description",""), severity=issue.get("severity","medium"), suggested_fix=issue.get("suggested_fix",""), legal_basis=issue.get("legal_basis","")))
                    is_sh = False
            recs = [self.generate_suggestions(c.clause_content, i.description) for i in issues]
            analyses.append(ClauseAnalysis(clause=c, issues=issues, is_compliant=is_comp, is_shariah_compliant=is_sh, recommendations=recs))
        score = (sum(1 for a in analyses if a.is_compliant) / len(analyses) * 100) if analyses else 0
        conflicts = [f"{a.clause.clause_type}: {i.description}" for a in analyses for i in a.issues if i.issue_type=="conflict"]
        sh_issues = [f"{a.clause.clause_type}: {i.description}" for a in analyses for i in a.issues if i.issue_type=="shariah_conflict"] if check_shariah else None
        return ContractAnalysis(contract_id=f"contract_{hash(contract_text)%10000}", missing_clauses=missing, analyzed_clauses=analyses, overall_compliance_score=score, jurisdiction_conflicts=conflicts, shariah_compliance_issues=sh_issues)

    def run(self, query: str) -> str:
        return self.agent.invoke({"input": query})

if __name__ == "__main__":
    agent = LegalAnalysisAgent()
    with open(sys.argv[1], 'r') as f:
        text = f.read()
    result = agent.analyze_full_contract(text, sys.argv[2] if len(sys.argv)>2 else "UAE", True)
    print(result)
