"""
Legal Analysis Agent - Main Initializer
This file contains the main initialization code for the Legal Analysis Agent,
including data ingestion, vector database setup, and agent configuration.
"""

import os
import glob
from typing import Dict, List, Any, Optional
import logging
from dotenv import load_dotenv

# LangChain imports
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_extraction_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
TEMPERATURE = (os.getenv("TEMPERATURE", 0))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./legal_knowledge_base")
LEGAL_DOCS_DIR = os.getenv("LEGAL_DOCS_DIR", "./legal_documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Define Pydantic models for structured outputs
class LegalClause(BaseModel):
    clause_id: str = Field(description="Unique identifier for the clause")
    clause_type: str = Field(description="Type of legal clause (e.g., termination, jurisdiction)")
    clause_content: str = Field(description="Full text content of the clause")
    section_number: Optional[str] = Field(None, description="Section or article number in the document")

class ClauseIssue(BaseModel):
    issue_type: str = Field(description="Type of issue (missing, conflict, ambiguous)")
    description: str = Field(description="Description of the legal issue")
    severity: str = Field(description="Severity level (high, medium, low)")
    suggested_fix: str = Field(description="Suggested text or approach to fix the issue")
    legal_basis: str = Field(description="Legal basis or reference for the issue identification")

class ClauseAnalysis(BaseModel):
    clause: LegalClause = Field(description="The analyzed clause")
    issues: List[ClauseIssue] = Field(description="List of identified issues")
    is_compliant: bool = Field(description="Whether the clause is compliant with requirements")
    is_shariah_compliant: Optional[bool] = Field(None, description="Whether the clause is Shariah compliant")
    recommendations: List[str] = Field(description="Overall recommendations for improving the clause")

class ContractAnalysis(BaseModel):
    contract_id: str = Field(description="Unique identifier for the contract")
    missing_clauses: List[str] = Field(description="Types of clauses that are missing")
    analyzed_clauses: List[ClauseAnalysis] = Field(description="Analysis of each clause")
    overall_compliance_score: float = Field(description="Overall compliance score from 0-100")
    jurisdiction_conflicts: List[str] = Field(description="Conflicts with jurisdiction rules")
    shariah_compliance_issues: Optional[List[str]] = Field(None, description="Shariah compliance issues if applicable")

class LegalAnalysisAgent:
    """Main class for the Legal Analysis Agent"""
    
    def __init__(self):
        """Initialize the Legal Analysis Agent with all necessary components"""
        logger.info("Initializing Legal Analysis Agent...")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Initialize vector database (if exists) or load data
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize agent with memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = self._create_agent()
        
        logger.info("Legal Analysis Agent initialized successfully")
    
    def _initialize_knowledge_base(self) -> Chroma:
        """Initialize or load the vector database with legal knowledge"""
        if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            logger.info(f"Loading existing knowledge base from {VECTOR_DB_DIR}")
            return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=self.embeddings)
        else:
            logger.info("Initializing new knowledge base with legal documents")
            return self._ingest_legal_knowledge()
    
    def _ingest_legal_knowledge(self) -> Chroma:
        """Ingest legal documents to create the knowledge base"""
        logger.info(f"Ingesting legal documents from {LEGAL_DOCS_DIR}")
        
        if not os.path.exists(LEGAL_DOCS_DIR):
            logger.error(f"Legal documents directory {LEGAL_DOCS_DIR} does not exist")
            raise FileNotFoundError(f"Directory {LEGAL_DOCS_DIR} not found")
        
        # Set up document loaders
        loaders = {
            "pdf": lambda path: PyPDFLoader(path),
            "docx": lambda path: Docx2txtLoader(path),
            "txt": lambda path: TextLoader(path)
        }
        
        # Process all documents in the directory
        all_docs = []
        for file_type, loader_func in loaders.items():
            file_pattern = os.path.join(LEGAL_DOCS_DIR, f"**/*.{file_type}")
            for file_path in glob.glob(file_pattern, recursive=True):
                try:
                    logger.info(f"Processing {file_path}")
                    loader = loader_func(file_path)
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["file_type"] = file_type
                        
                        # Extract jurisdiction and document type from folder structure or filename
                        parts = file_path.split(os.sep)
                        if len(parts) > 1:
                            doc.metadata["jurisdiction"] = parts[-2] if len(parts) > 2 else "unknown"
                            doc.metadata["document_type"] = self._infer_document_type(file_path)
                    
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(all_docs)} documents. Splitting into chunks...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(all_docs)
        
        logger.info(f"Created {len(chunks)} chunks. Creating vector database...")
        
        # Create and persist the vector database
        knowledge_base = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        knowledge_base.persist()
        
        logger.info(f"Knowledge base created with {len(chunks)} chunks")
        return knowledge_base
    
    def _infer_document_type(self, file_path: str) -> str:
        """Infer document type from filename or content"""
        filename = os.path.basename(file_path).lower()
        
        if "template" in filename:
            return "template"
        elif "regulation" in filename or "law" in filename:
            return "regulation"
        elif "case" in filename or "precedent" in filename:
            return "precedent"
        elif "guide" in filename or "manual" in filename:
            return "guide"
        elif "shariah" in filename:
            return "shariah_standard"
        else:
            return "other"

    def _create_tools(self) -> List[Tool]:
        """Create the tools for the agent"""
        return [
            Tool(
                name="ClauseExtractor",
                func=self.extract_clauses,
                description="Extracts and identifies clauses from legal documents. Input should be the text content of the document."
            ),
            Tool(
                name="GapDetector",
                func=self.detect_gaps,
                description="Identifies missing clauses by comparing to templates. Input should be the document type and jurisdiction."
            ),
            Tool(
                name="ComplianceChecker",
                func=self.check_compliance,
                description="Checks if clauses comply with jurisdiction requirements. Input should be the clause text and jurisdiction."
            ),
            Tool(
                name="ShariahComplianceChecker",
                func=self.check_shariah_compliance,
                description="Checks if clauses comply with Shariah law. Input should be the clause text."
            ),
            Tool(
                name="SuggestionGenerator",
                func=self.generate_suggestions,
                description="Creates suggested revisions for problematic clauses. Input should be the clause text and the identified issue."
            ),
            Tool(
                name="KnowledgeBaseSearch",
                func=self.search_knowledge_base,
                description="Searches the legal knowledge base for relevant information. Input should be the search query."
            ),
            Tool(
                name="FullContractAnalyzer",
                func=self.analyze_full_contract,
                description="Analyzes a complete contract document. Input should be the contract text, jurisdiction, and whether Shariah compliance is required."
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor"""
        # Create a system prompt template
        system_prompt = """You are a legal analysis agent specializing in contract review. 
        Your expertise includes various legal systems and Shariah law compliance.
        
        You help users by:
        1. Analyzing contracts for completeness and compliance
        2. Identifying missing clauses and legal conflicts
        3. Detecting jurisdiction-specific issues
        4. Checking Shariah compliance where relevant
        5. Suggesting specific improvements to legal language
        
        Provide thoughtful, detailed analysis based on legal principles and the specific jurisdiction requirements.
        Always explain your reasoning and the legal basis for your suggestions.
        
        When analyzing contracts, first extract clauses, then check each one systematically against requirements.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    # Tool implementation methods
    
    def extract_clauses(self, document_text: str) -> List[LegalClause]:
        """Extract clauses from document text"""
        logger.info("Extracting clauses from document")
        
        # Define extraction schema
        schema = {
            "properties": {
                "clauses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "clause_id": {"type": "string"},
                            "clause_type": {"type": "string"},
                            "clause_content": {"type": "string"},
                            "section_number": {"type": "string"}
                        },
                        "required": ["clause_type", "clause_content"]
                    }
                }
            },
            "required": ["clauses"]
        }
        
        # Create extraction chain
        extraction_chain = create_extraction_chain(schema, self.llm)
        
        # Run extraction
        result = extraction_chain.invoke({"input": document_text})

        
        # Convert to Pydantic models
        clauses = []
        if "clauses" in result:
            for i, clause_data in enumerate(result["clauses"]):
                # Set default clause_id if not provided
                if "clause_id" not in clause_data:
                    clause_data["clause_id"] = f"clause_{i+1}"
                
                clauses.append(LegalClause(**clause_data))
        
        logger.info(f"Extracted {len(clauses)} clauses")
        return clauses
    
    def detect_gaps(self, document_text: str, document_type: str, jurisdiction: str) -> List[str]:
        """Detect missing clauses by comparing to templates"""
        logger.info(f"Detecting gaps for {document_type} in {jurisdiction}")
        
        # Define the prompt
        template = """
        For a {document_type} in {jurisdiction}, identify which standard clauses are typically required.
        Then provide a list of clause types that appear to be missing from the contract.
        
        this is the document that needs to be analyzed:
        {document_text}
        
        Required clauses typically include:
        - Parties identification
        - Definitions
        - Scope of work/services
        - Payment terms
        - Term and termination
        - Representations and warranties
        - Limitation of liability
        - Indemnification
        - Confidentiality
        - Governing law and jurisdiction
        - Dispute resolution
        - Force majeure
        - Assignment
        - Entire agreement
        - Notices
        
        Consider jurisdiction-specific requirements for {jurisdiction}.
        
        Return only the list of missing clause types.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["document_type", "jurisdiction"]
        )
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Run chain
        result = chain.run(document_type=document_type, jurisdiction=jurisdiction)
        
        # Parse result (assuming result is a newline-separated list)
        missing_clauses = [clause.strip() for clause in result.split('\n') if clause.strip()]
        
        logger.info(f"Detected {len(missing_clauses)} missing clauses")
        return missing_clauses
    
    def check_compliance(self, clause_text: str, jurisdiction: str) -> Dict[str, Any]:
        """Check if clause complies with jurisdiction requirements"""
        logger.info(f"Checking compliance for clause in {jurisdiction}")
        
        # Retrieve relevant knowledge
        retriever = self.knowledge_base.as_retriever(
            search_kwargs={"k": 5, "filter": {"jurisdiction": jurisdiction}}
        )
        
        # Define the prompt
        template = """
        Analyze the following legal clause for compliance with {jurisdiction} laws and regulations:
        
        CLAUSE: {clause_text}
        
        
        Based on your legal knowledge and the following context from the legal knowledge base, identify:
        1. Is this clause compliant with {jurisdiction} laws?
        2. Are there any specific legal conflicts or issues?
        3. What is the legal basis for any identified issues?
        4. How severe are these issues (high, medium, low)?
        
        CONTEXT FROM KNOWLEDGE BASE:
        {context}
        
        Provide a structured analysis with specific issues and their legal basis.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create document chain
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        
        # Run analysis
        result = retrieval_chain.invoke({
            "clause_text": clause_text,
            "jurisdiction": jurisdiction
        })
        
        logger.info("Compliance check completed")
        return result
    
    def check_shariah_compliance(self, clause_text: str) -> Dict[str, Any]:
        """Check if clause complies with Shariah law"""
        logger.info("Checking Shariah compliance for clause")
        
        # Retrieve relevant knowledge
        retriever = self.knowledge_base.as_retriever(
            search_kwargs={"k": 5, "filter": {"document_type": "shariah_standard"}}
        )
        
        # Define the prompt
        template = """
        Analyze the following legal clause for compliance with Shariah law:
        
        CLAUSE: {clause_text}
        
        Based on your knowledge of Islamic finance and the following context from the Shariah standards knowledge base, identify:
        1. Is this clause Shariah-compliant?
        2. Are there any specific Shariah compliance issues?
        3. What is the basis in Islamic jurisprudence for any identified issues?
        4. How severe are these issues (high, medium, low)?
        
        CONTEXT FROM KNOWLEDGE BASE:
        {context}
        
        Consider principles like riba (interest), gharar (uncertainty), maysir (gambling), and underlying contract types (murabaha, ijara, etc.).
        
        Provide a structured analysis with specific issues and their basis in Islamic jurisprudence.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create document chain
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        
        # Run analysis
        result = retrieval_chain.invoke({
            "clause_text": clause_text
        })
        
        logger.info("Shariah compliance check completed")
        return result
    
    def generate_suggestions(self, clause_text: str, issue_description: str) -> str:
        """Generate suggestions for improving problematic clauses"""
        logger.info("Generating suggestions for clause")
        
        # Define the prompt
        template = """
        You need to revise the following legal clause to fix this issue:
        
        CLAUSE: {clause_text}
        
        ISSUE: {issue_description}
        
        Please provide:
        1. A revised version of the clause that addresses the issue
        2. An explanation of the changes made and why they address the issue
        
        Your suggestions should maintain the legal intent of the original clause while fixing the identified problems.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["clause_text", "issue_description"]
        )
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Run chain
        result = chain.run(clause_text=clause_text, issue_description=issue_description)
        
        logger.info("Suggestion generated")
        return result
    
    def search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search the legal knowledge base for relevant information"""
        logger.info(f"Searching knowledge base for: {query}")
        
        # Perform vector search
        docs = self.knowledge_base.similarity_search(query, k=5)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance": "high"  # In a real implementation, you would have actual relevance scores
            })
        
        logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def analyze_full_contract(self, contract_text: str, jurisdiction: str, check_shariah: bool = False) -> ContractAnalysis:
        """Analyze a complete contract document"""
        logger.info(f"Analyzing full contract for {jurisdiction}, Shariah check: {check_shariah}")
        
        # Extract clauses
        clauses = self.extract_clauses(contract_text)
        
        # Detect gaps
        document_type = "contract"  # This could be inferred in a more sophisticated implementation
        missing_clauses = self.detect_gaps(contract_text, document_type, jurisdiction)
        
        # Analyze each clause
        analyzed_clauses = []
        for clause in clauses:
            # Check jurisdiction compliance
            compliance_result = self.check_compliance(clause.clause_content, jurisdiction)
            
            # Check Shariah compliance if requested
            shariah_result = None
            if check_shariah:
                shariah_result = self.check_shariah_compliance(clause.clause_content)
            
            # Create issues list
            issues = []
            is_compliant = True
            is_shariah_compliant = True if not check_shariah else False
            
            # Process compliance issues
            if "issues" in compliance_result:
                for issue in compliance_result["issues"]:
                    issues.append(
                        ClauseIssue(
                            issue_type="conflict",
                            description=issue["description"],
                            severity=issue.get("severity", "medium"),
                            suggested_fix=issue.get("suggested_fix", ""),
                            legal_basis=issue.get("legal_basis", "")
                        )
                    )
                    is_compliant = False
            
            # Process Shariah compliance issues
            if check_shariah and shariah_result and "issues" in shariah_result:
                for issue in shariah_result["issues"]:
                    issues.append(
                        ClauseIssue(
                            issue_type="shariah_conflict",
                            description=issue["description"],
                            severity=issue.get("severity", "medium"),
                            suggested_fix=issue.get("suggested_fix", ""),
                            legal_basis=issue.get("legal_basis", "")
                        )
                    )
                    is_shariah_compliant = False
            
            # Get recommendations
            recommendations = []
            if not is_compliant or (check_shariah and not is_shariah_compliant):
                for issue in issues:
                    suggestion = self.generate_suggestions(clause.clause_content, issue.description)
                    recommendations.append(suggestion)
            
            # Create clause analysis
            clause_analysis = ClauseAnalysis(
                clause=clause,
                issues=issues,
                is_compliant=is_compliant,
                is_shariah_compliant=is_shariah_compliant if check_shariah else None,
                recommendations=recommendations
            )
            
            analyzed_clauses.append(clause_analysis)
        
        # Calculate overall compliance score
        total_clauses = len(clauses)
        compliant_clauses = sum(1 for analysis in analyzed_clauses if analysis.is_compliant)
        compliance_score = (compliant_clauses / total_clauses * 100) if total_clauses > 0 else 0
        
        # Extract jurisdiction conflicts
        jurisdiction_conflicts = []
        for analysis in analyzed_clauses:
            for issue in analysis.issues:
                if issue.issue_type == "conflict":
                    conflict = f"{analysis.clause.clause_type}: {issue.description}"
                    jurisdiction_conflicts.append(conflict)
        
        # Extract Shariah compliance issues
        shariah_compliance_issues = None
        if check_shariah:
            shariah_compliance_issues = []
            for analysis in analyzed_clauses:
                for issue in analysis.issues:
                    if issue.issue_type == "shariah_conflict":
                        conflict = f"{analysis.clause.clause_type}: {issue.description}"
                        shariah_compliance_issues.append(conflict)
        
        # Create contract analysis
        contract_analysis = ContractAnalysis(
            contract_id=f"contract_{hash(contract_text) % 10000}",  # Simple hash-based ID
            missing_clauses=missing_clauses,
            analyzed_clauses=analyzed_clauses,
            overall_compliance_score=compliance_score,
            jurisdiction_conflicts=jurisdiction_conflicts,
            shariah_compliance_issues=shariah_compliance_issues
        )
        
        logger.info("Full contract analysis completed")
        return contract_analysis
    
    def run(self, query: str) -> str:
        """Run the agent on a user query"""
        return self.agent.invoke({"input": query})


if __name__ == "__main__":
    agent = LegalAnalysisAgent()
    

    result = agent.run("""
    Please analyze this contract clause:
    
    "In the event of default, the Borrower shall pay interest at a rate of 5% per month on the outstanding balance until full payment is made."
    
    This contract will be used in the UAE.
    """)
    
    print("\nFinal Result:")
    print(result)