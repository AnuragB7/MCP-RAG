import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging
from config import Config
import traceback
import os

logger = logging.getLogger(__name__)

class RAGMultiDocumentAgent:
    """RAG-enabled agent for intelligent document retrieval and analysis"""
    
    def __init__(self, model_name="gpt-4o", server_script_path="./mcp_server.py"):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Ensure server script exists
        if not os.path.exists(server_script_path):
            raise ValueError(f"MCP server script not found: {server_script_path}")
            
        self.model = ChatOpenAI(
            model_name=Config.MODEL_NAME,
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.BASE_URL,
            temperature=0.1,
            timeout=120  # 2 minute timeout
        )
        
        self.server_params = StdioServerParameters(
            command=sys.executable,  # Use current Python interpreter
            args=[server_script_path],
            env=dict(os.environ)  # Pass current environment
        )
        logger.info(f"Initialized RAG agent with model: {model_name}")
    
    async def _execute_with_session(self, operation_func, *args, **kwargs):
        """Execute operation with proper session management and error handling"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Executing operation (attempt {attempt + 1}/{max_retries})")
                
                async with stdio_client(self.server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        # Initialize the connection
                        await session.initialize()
                        
                        # Load MCP tools
                        tools = await load_mcp_tools(session)
                        logger.info(f"Loaded {len(tools)} MCP tools")
                        
                        # Create agent
                        agent = create_react_agent(self.model, tools)
                        
                        # Execute operation
                        return await operation_func(agent, *args, **kwargs)
                        
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    logger.error(traceback.format_exc())
                    raise
    
    async def upload_and_index_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Upload documents and create vector embeddings"""
        
        async def _upload_operation(agent, file_paths):
            prompt = f"""
            Please upload and process these documents with RAG capabilities: {file_paths}
            
            Use the upload_documents_with_rag tool to:
            1. Extract content from all documents (handle large files appropriately)
            2. Create vector embeddings for semantic search
            3. Provide a summary of the RAG indexing process
            4. Report any processing issues or large file handling
            
            After uploading, use get_rag_document_summary to confirm successful processing.
            """
            
            return await agent.ainvoke({
                "messages": [{"role": "user", "content": prompt}]
            })
        
        return await self._execute_with_session(_upload_operation, file_paths)
    
    async def rag_query(self, question: str) -> Dict[str, Any]:
        """Ask questions using RAG retrieval"""
        
        async def _query_operation(agent, question):
            prompt = f"""
            Please answer this question using RAG (Retrieval-Augmented Generation): {question}
            
            Process:
            1. First, use retrieve_context_for_query to get relevant context from the vector database
            2. Use semantic_search_documents to find the most relevant chunks if needed for more details
            3. Analyze the retrieved content carefully
            4. Provide a comprehensive answer citing specific sources and their file types
            5. Include similarity scores when available
            6. If the retrieved context doesn't fully answer the question, acknowledge the limitations
            
            Question: {question}
            
            Make sure to:
            - Explain which documents contributed to your answer
            - Include confidence levels based on similarity scores
            - Distinguish between information from different file types (PDF, Excel, DOCX, CSV)
            - Be clear about what information is missing if the question can't be fully answered
            """
            
            return await agent.ainvoke({
                "messages": [{"role": "user", "content": prompt}]
            })
        
        return await self._execute_with_session(_query_operation, question)
    
    async def analyze_document_collection(self) -> Dict[str, Any]:
        """Analyze the uploaded document collection"""
        
        async def _analyze_operation(agent):
            prompt = """
            Please provide a comprehensive analysis of the uploaded document collection:
            
            1. Use analyze_document_collection to get collection statistics
            2. Use get_rag_document_summary for detailed document information
            3. Analyze the file type distribution
            4. Comment on large file handling
            5. Assess the overall scope and diversity of the documents
            6. Provide insights about the embedding and chunking process
            
            Present a clear summary of what's in the document collection and how well it's been processed.
            """
            
            return await agent.ainvoke({
                "messages": [{"role": "user", "content": prompt}]
            })
        
        return await self._execute_with_session(_analyze_operation)
    
    async def multi_document_research(self, research_question: str) -> Dict[str, Any]:
        """Perform comprehensive research across all documents"""
        
        async def _research_operation(agent, research_question):
            prompt = f"""
            I need comprehensive research to answer this question: {research_question}
            
            Please conduct thorough research by:
            1. Using retrieve_context_for_query to get initial relevant context
            2. Using semantic_search_documents with different search terms related to the question
            3. Analyzing content from different document types and sources
            4. Synthesizing findings from multiple retrieval attempts
            5. Cross-referencing information between documents
            6. Providing a well-structured research report with:
               - Executive summary
               - Key findings from each relevant document
               - Supporting evidence with source citations
               - Analysis of data patterns across document types
               - Confidence assessment based on source reliability
               - Gaps in available information
            
            Research Question: {research_question}
            
            Be thorough and academic in your approach. Use multiple search queries to ensure comprehensive coverage.
            """
            
            return await agent.ainvoke({
                "messages": [{"role": "user", "content": prompt}]
            })
        
        return await self._execute_with_session(_research_operation, research_question)
    
    async def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the RAG store"""
        
        async def _clear_operation(agent):
            prompt = """
            Please clear all documents from the RAG document store using the clear_document_store tool.
            Confirm the operation was successful.
            """
            
            return await agent.ainvoke({
                "messages": [{"role": "user", "content": prompt}]
            })
        
        return await self._execute_with_session(_clear_operation)

# Example usage
async def main():
    try:
        agent = RAGMultiDocumentAgent()
        
        # Example file paths
        files = [
            "sample_report.pdf",
            "data_analysis.xlsx", 
            "meeting_notes.docx",
            "survey_results.csv"
        ]
        
        # Upload and index documents
        print("Uploading and indexing documents...")
        upload_result = await agent.upload_and_index_documents(files)
        print("Upload result:", upload_result)
        
        # Perform RAG query
        print("\nPerforming RAG query...")
        query_result = await agent.rag_query("What are the main insights from the data?")
        print("Query result:", query_result)
        
        # Analyze collection
        print("\nAnalyzing document collection...")
        analysis_result = await agent.analyze_document_collection()
        print("Analysis result:", analysis_result)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
