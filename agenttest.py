import json
import os
import glob
import sys
import logging
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import create_extraction_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load env & configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

# Config
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
MODEL_NAME        = os.getenv("MODEL_NAME", "gpt-4-turbo")
TEMPERATURE       = float(os.getenv("TEMPERATURE", "0"))
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
VECTOR_DB_DIR     = os.getenv("VECTOR_DB_DIR", "./legal_knowledge_base")
LEGAL_DOCS_DIR    = os.getenv("LEGAL_DOCS_DIR", "./legal_documents")
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "200"))

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models you already defined (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model & parser for structured gap-detection
# ─────────────────────────────────────────────────────────────────────────────
class GapDetectionOutput(BaseModel):
    missing_clauses: List[str]

# ─────────────────────────────────────────────────────────────────────────────
# The fully refactored agent
# ─────────────────────────────────────────────────────────────────────────────
class LegalAnalysisAgent:
    def __init__(self):
        logger.info("Initializing LegalAnalysisAgent…")
        self.llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE,
                              streaming=False,
                              callbacks=[StreamingStdOutCallbackHandler()])
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.knowledge_base = self._load_or_create_vector_db()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tools = self._create_tools()
        self.agent = self._create_agent_executor()
        logger.info("Initialization complete.")

    def _load_or_create_vector_db(self) -> Chroma:
        if os.path.isdir(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=self.embeddings)
        # else ingest
        if not os.path.isdir(LEGAL_DOCS_DIR):
            raise FileNotFoundError(f"{LEGAL_DOCS_DIR} not found")
        all_docs, loaders = [], {
            "pdf": PyPDFLoader, "docx": Docx2txtLoader, "txt": TextLoader
        }
        for ext, Loader in loaders.items():
            for path in glob.glob(os.path.join(LEGAL_DOCS_DIR, f"**/*.{ext}"), recursive=True):
                docs = Loader(path).load()
                for d in docs:
                    d.metadata["source"] = path
                all_docs.extend(docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(all_docs)
        db = Chroma.from_documents(chunks, embedding=self.embeddings, persist_directory=VECTOR_DB_DIR)
        db.persist()
        return db

    def _create_tools(self):
        return [
            Tool("ClauseExtractor",          func=self.extract_clauses,          description="Extract clauses"),
            Tool("GapDetector",              func=self.detect_gaps,               description="Detect missing clauses"),
            Tool("ComplianceChecker",        func=self.check_compliance,          description="Check compliance"),
            Tool("ShariahComplianceChecker", func=self.check_shariah_compliance,  description="Shariah compliance"),
            Tool("SuggestionGenerator",      func=self.generate_suggestions,      description="Suggest fixes"),
            Tool("KnowledgeBaseSearch",      func=self.search_knowledge_base,     description="Search KB"),
            Tool("FullContractAnalyzer",     func=self.analyze_full_contract,     description="Analyze full contract")
        ]

    def _create_agent_executor(self) -> AgentExecutor:
        system = "You are a legal analysis agent. Use the provided tools when needed."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def extract_clauses(self, document_text: str) -> List[LegalClause]:
        schema = {
            "properties": {
                "clauses": {
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "clause_id":{"type":"string"},
                            "clause_type":{"type":"string"},
                            "clause_content":{"type":"string"},
                            "section_number":{"type":"string"}
                        },
                        "required":["clause_type","clause_content"]
                    }
                }
            },
            "required":["clauses"]
        }
        chain = create_extraction_chain(schema, self.llm)
        result = chain.invoke({"input": document_text})
        data = result.get("clauses", []) if isinstance(result, dict) else []
        clauses = []
        for idx, c in enumerate(data):
            if "clause_id" not in c:
                c["clause_id"] = f"clause_{idx+1}"
            clauses.append(LegalClause(**c))
        return clauses

    def detect_gaps(self, document_text: str, document_type: str, jurisdiction: str) -> List[str]:
        # Use PydanticOutputParser for strict JSON
        parser = PydanticOutputParser(pydantic_object=GapDetectionOutput)
        prompt = PromptTemplate.from_template(
            "You are a legal expert reviewing a {document_type} in {jurisdiction}."
            " Identify any missing or underdeveloped clauses.  Return exactly:\n\n"
            "{format_instructions}\n\n"
            "Document:\n{document_text}"
        ).partial(format_instructions=parser.get_format_instructions())

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "document_text": document_text,
            "document_type": document_type,
            "jurisdiction": jurisdiction
        })

        raw = response.get("text", "") if isinstance(response, dict) else str(response)
        try:
            parsed = parser.parse(raw)
            return parsed.missing_clauses
        except Exception as e:
            logger.error("Gap parsing failed, raw output:\n%s\nError: %s", raw, e)
            # Fallback: simple line-split
            return [line.strip("-•* ") for line in raw.splitlines() if line.strip()]

    def check_compliance(self, clause_text: str, jurisdiction: str) -> Dict[str, Any]:
        retr = self.knowledge_base.as_retriever(search_kwargs={"k":5, "filter":{"jurisdiction": jurisdiction}})
        prompt = ChatPromptTemplate.from_template(
            "Analyze compliance of: {clause_text} wrt {jurisdiction}. Context: {context}"
        )
        chain = create_retrieval_chain(retr, create_stuff_documents_chain(self.llm, prompt))
        result = chain.invoke({"clause_text": clause_text, "jurisdiction": jurisdiction})
        if "issues" not in result:
            result["issues"] = []
        return result

    def check_shariah_compliance(self, clause_text: str) -> Dict[str, Any]:
        retr = self.knowledge_base.as_retriever(search_kwargs={"k":5, "filter":{"document_type":"shariah_standard"}})
        prompt = ChatPromptTemplate.from_template(
            "Analyze Shariah compliance of: {clause_text}. Context: {context}"
        )
        chain = create_retrieval_chain(retr, create_stuff_documents_chain(self.llm, prompt))
        result = chain.invoke({"clause_text": clause_text})
        if "issues" not in result:
            result["issues"] = []
        return result

    def generate_suggestions(self, clause_text: str, issue_description: str) -> str:
        prompt = PromptTemplate(
            template="Revise clause: {clause_text} to fix: {issue_description}",
            input_variables=["clause_text","issue_description"]
        )
        resp = LLMChain(llm=self.llm, prompt=prompt).invoke({
            "clause_text": clause_text,
            "issue_description": issue_description
        })
        return (resp.get("text") or resp.get("output") or str(resp)).strip()

    def search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        docs = self.knowledge_base.similarity_search(query, k=5)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    def analyze_full_contract(self, contract_text: str, jurisdiction: str, check_shariah: bool=False) -> ContractAnalysis:
        # 1. Extract
        clauses = self.extract_clauses(contract_text)

        # 2. Gaps
        missing = self.detect_gaps(contract_text, "contract", jurisdiction)

        # 3. Per-clause analysis
        analyses: List[ClauseAnalysis] = []
        for c in clauses:
            comp = self.check_compliance(c.clause_content, jurisdiction)
            sh   = self.check_shariah_compliance(c.clause_content) if check_shariah else {"issues": []}

            issues: List[ClauseIssue] = []
            for issue in comp.get("issues", []):
                issues.append(ClauseIssue(
                    issue_type="conflict",
                    description=issue.get("description",""),
                    severity=issue.get("severity","medium"),
                    suggested_fix=issue.get("suggested_fix",""),
                    legal_basis=issue.get("legal_basis","")
                ))
            for issue in sh.get("issues", []):
                issues.append(ClauseIssue(
                    issue_type="shariah_conflict",
                    description=issue.get("description",""),
                    severity=issue.get("severity","medium"),
                    suggested_fix=issue.get("suggested_fix",""),
                    legal_basis=issue.get("legal_basis","")
                ))

            recommendations = [
                self.generate_suggestions(c.clause_content, iss.description)
                for iss in issues
            ]

            analyses.append(ClauseAnalysis(
                clause=c,
                issues=issues,
                is_compliant=not any(i.issue_type=="conflict" for i in issues),
                is_shariah_compliant=(not any(i.issue_type=="shariah_conflict" for i in issues)) if check_shariah else None,
                recommendations=recommendations
            ))

        # 4. Score & conflicts
        total = len(analyses) or 1
        compliant_count = sum(1 for a in analyses if a.is_compliant)
        score = round((compliant_count/total)*100, 2)

        juris_conflicts = [
            f"{a.clause.clause_type}: {i.description}"
            for a in analyses for i in a.issues if i.issue_type=="conflict"
        ]
        shariah_conflicts = [
            f"{a.clause.clause_type}: {i.description}"
            for a in analyses for i in a.issues if i.issue_type=="shariah_conflict"
        ] if check_shariah else None

        return ContractAnalysis(
            contract_id=f"contract_{hash(contract_text)%10000}",
            missing_clauses=missing,
            analyzed_clauses=analyses,
            overall_compliance_score=score,
            jurisdiction_conflicts=juris_conflicts,
            shariah_compliance_issues=shariah_conflicts
        )

    def run(self, query: str) -> str:
        return self.agent.invoke({"input": query})

if __name__ == "__main__":
    agent = LegalAnalysisAgent()
    try:
        text = open(sys.argv[1]).read()
        jur  = sys.argv[2] if len(sys.argv)>2 else "UAE"
        chk  = (len(sys.argv)>3 and sys.argv[3].lower()=="true")
        result: ContractAnalysis = agent.analyze_full_contract(text, jur, chk)
        print(result.json(indent=2))
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)
