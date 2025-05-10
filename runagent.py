#!/usr/bin/env python3
"""
Simple CLI runner for the Legal Analysis Agent
"""

import sys
import os
from dotenv import load_dotenv
from agent import LegalAnalysisAgent

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the agent
    print("Initializing Legal Analysis Agent...")
    agent = LegalAnalysisAgent()
    print("Agent initialized successfully!")
    
    if len(sys.argv) > 1:
        # Process a file if provided
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return
            
        print(f"Analyzing contract file: {file_path}")
        
        # Read the file
        with open(file_path, 'r') as f:
            contract_text = f.read()
            
        # Default jurisdiction to UAE if not specified
        jurisdiction = "UAE"
        if len(sys.argv) > 2:
            jurisdiction = sys.argv[2]
            
        # Default Shariah check to True if not specified  
        check_shariah = True
        if len(sys.argv) > 3:
            check_shariah = sys.argv[3].lower() in ['true', 'yes', '1']
            
        # Analyze the contract
        result = agent.analyze_full_contract(contract_text, jurisdiction, check_shariah)
        
        # Print results
        print("\n=== CONTRACT ANALYSIS RESULTS ===")
        print(f"Contract ID: {result.contract_id}")
        print(f"Overall Compliance Score: {result.overall_compliance_score:.2f}%")
        
        if result.missing_clauses:
            print("\n=== MISSING CLAUSES ===")
            for clause in result.missing_clauses:
                print(f"- {clause}")
                
        if result.jurisdiction_conflicts:
            print("\n=== JURISDICTION CONFLICTS ===")
            for conflict in result.jurisdiction_conflicts:
                print(f"- {conflict}")
                
        if result.shariah_compliance_issues:
            print("\n=== SHARIAH COMPLIANCE ISSUES ===")
            for issue in result.shariah_compliance_issues:
                print(f"- {issue}")
                
        print("\n=== DETAILED CLAUSE ANALYSIS ===")
        for analysis in result.analyzed_clauses:
            print(f"\nClause Type: {analysis.clause.clause_type}")
            print(f"Compliant: {'✅' if analysis.is_compliant else '❌'}")
            if analysis.is_shariah_compliant is not None:
                print(f"Shariah Compliant: {'✅' if analysis.is_shariah_compliant else '❌'}")
            
            if analysis.issues:
                print("Issues:")
                for issue in analysis.issues:
                    print(f"  - {issue.issue_type} ({issue.severity}): {issue.description}")
                    
            if analysis.recommendations:
                print("Recommendations:")
                for i, rec in enumerate(analysis.recommendations):
                    print(f"  {i+1}. {rec[:100]}...")  # Show first 100 chars
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            result = agent.run(query)
            print("\n=== RESULT ===")
            print(result)
            print("==============")

if __name__ == "__main__":
    main()