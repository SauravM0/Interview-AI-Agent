"""
RAG (Retrieval-Augmented Generation) Processor for candidate resume analysis.
Handles document processing, vector storage, and natural language querying.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from pathlib import Path

# LangChain imports
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
        UnstructuredFileLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.llms import Ollama
    from langchain.schema import Document
except ImportError as e:
    logging.error(f"Failed to import LangChain components: {e}")
    raise

logger = logging.getLogger(__name__)

class RAGProcessor:
    """
    A class to handle document processing and retrieval using RAG (Retrieval-Augmented Generation).
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG processor with necessary components.
        
        Args:
            persist_directory (str): Directory to persist the Chroma database
        """
        self.persist_directory = os.path.abspath(persist_directory)
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the vector store
        self._initialize_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize the HuggingFace embeddings model."""
        try:
            logger.info("Initializing HuggingFace embeddings...")
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                },
                show_progress=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize or load the Chroma vector store."""
        try:
            logger.info(f"Initializing Chroma vector store at {self.persist_directory}")
            
            # Ensure the directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="resumes"
            )
            
            # Create a retriever with MMR (Maximal Marginal Relevance) for better diversity
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 3,  # Number of documents to return
                    "fetch_k": 20,  # Number of documents to fetch before MMR
                    "lambda_mult": 0.5  # Diversity parameter (0-1)
                }
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Try to create a new vector store if loading fails
            logger.info("Attempting to create a new vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="resumes"
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            )
    
    def _load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on its file extension.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of document chunks
        """
        try:
            logger.info(f"Loading document: {file_path}")
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Try different loaders based on file extension
            try:
                if file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                elif file_ext in ['.docx', '.doc']:
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                elif file_ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                else:
                    # Try unstructured loader for other formats
                    loader = UnstructuredFileLoader(file_path)
                    documents = loader.load()
            except Exception as load_error:
                logger.warning(f"Error with specialized loader for {file_path}: {load_error}")
                logger.info("Falling back to generic text loader...")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    documents = [Document(page_content=text, metadata={"source": file_path})]
                except Exception as fallback_error:
                    logger.error(f"Failed to load document {file_path} with fallback: {fallback_error}")
                    raise
            
            if not documents:
                logger.warning(f"No content loaded from {file_path}")
                return []
            
            # Log basic info about the loaded documents
            logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
            raise
    
    def process_document(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """
        Process a document and add it to the vector store.
        
        Args:
            file_path (str): Path to the document file
            metadata (Dict, optional): Additional metadata for the document
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            logger.info(f"Processing document: {file_path}")
            
            # Ensure metadata is a dictionary
            if metadata is None:
                metadata = {}
            
            # Add file info to metadata if not already present
            if 'source' not in metadata:
                metadata['source'] = os.path.basename(file_path)
            
            # Load and split the document
            chunks = self._load_document(file_path)
            
            if not chunks:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update(metadata)
            
            try:
                # Add to vector store
                if hasattr(self.vector_store, 'add_documents'):
                    self.vector_store.add_documents(chunks)
                    
                    # Ensure changes are persisted
                    if hasattr(self.vector_store, 'persist'):
                        self.vector_store.persist()
                else:
                    logger.error("Vector store doesn't support add_documents method")
                    return False
                
                logger.info(f"Successfully processed and added {len(chunks)} chunks from {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                # Try to reinitialize the vector store and retry once
                try:
                    logger.info("Attempting to reinitialize vector store and retry...")
                    self._initialize_vector_store()
                    self.vector_store.add_documents(chunks)
                    if hasattr(self.vector_store, 'persist'):
                        self.vector_store.persist()
                    logger.info("Successfully added documents after reinitialization")
                    return True
                except Exception as retry_error:
                    logger.error(f"Failed to add documents after retry: {retry_error}")
                    return False
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
            return False
    
    def process_text(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """
        Process raw text and add it to the vector store.
        
        Args:
            text (str): The text to process
            metadata (Dict, optional): Additional metadata for the text
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            if not text.strip():
                logger.warning("Empty text provided for processing")
                return False
            
            # Create a document
            doc = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # Initialize QA chain with Ollama
            llm = Ollama(
                model="llama3",  # or any other model you have downloaded with Ollama
                temperature=0.1,
                num_predict=1000
            )
            
            # Add to vector store
            self.vector_store.add_documents([doc])
            self.vector_store.persist()
            
            logger.info("Successfully processed and added text to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return False
    
    def query(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query (str): The query string
            k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with scores
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty or invalid query provided")
                return []
            
            logger.info(f"Executing query: {query}")
            
            # Ensure vector store is initialized
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                logger.error("Vector store not initialized")
                self._initialize_vector_store()
            
            # Get relevant documents with similarity scores
            try:
                if hasattr(self.vector_store, 'similarity_search_with_score'):
                    docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
                else:
                    # Fallback to using the retriever
                    logger.warning("similarity_search_with_score not available, using retriever")
                    docs = self.retriever.get_relevant_documents(query)
                    docs_and_scores = [(doc, 1.0) for doc in docs[:k]]
                    
                # Format results
                results = []
                for doc, score in docs_and_scores:
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(score) if hasattr(score, '__float__') else 1.0
                    }
                    results.append(result)
                
                logger.info(f"Found {len(results)} results for query")
                return results
                
            except Exception as search_error:
                logger.error(f"Error during similarity search: {search_error}", exc_info=True)
                # Try fallback search method
                try:
                    logger.info("Attempting fallback search method...")
                    docs = self.vector_store.similarity_search(query, k=k)
                    return [{
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': 1.0
                    } for doc in docs]
                except Exception as fallback_error:
                    logger.error(f"Fallback search also failed: {fallback_error}")
                    return []
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}", exc_info=True)
            return []
    
    def get_candidate_insights(self, query: str) -> str:
        """
        Get insights about candidates based on a natural language query.
        
        Args:
            query (str): Natural language query about candidates
            
        Returns:
            str: Formatted response with insights
        """
        try:
            logger.info(f"Getting candidate insights for query: {query}")
            
            # Ensure we have a retriever
            if not hasattr(self, 'retriever') or self.retriever is None:
                logger.error("Retriever not initialized")
                self._initialize_vector_store()
                if not hasattr(self, 'retriever') or self.retriever is None:
                    return "Error: Could not initialize the document retriever."
            
            # Initialize QA chain if not already done
            if not hasattr(self, 'qa_chain') or self.qa_chain is None:
                logger.info("Initializing QA chain...")
                try:
                    # Try to use Ollama
                    llm = Ollama(
                        model="llama3",  # or any other model you have downloaded with Ollama
                        temperature=0.1,  # Lower temperature for more focused answers
                        num_predict=1000
                    )
                    self.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=self.retriever,
                        return_source_documents=True
                    )
                    logger.info("QA chain initialized with Ollama")
                except Exception as chain_error:
                    logger.error(f"Error initializing QA chain: {chain_error}")
                    self.qa_chain = None
            
            # If we have a QA chain, use it
            if self.qa_chain:
                try:
                    result = self.qa_chain({"query": query})
                    
                    # Format the response
                    response = f"Query: {query}\n\n"
                    response += f"Answer: {result.get('result', 'No answer found')}\n\n"
                    
                    # Add source documents if available
                    if 'source_documents' in result and result['source_documents']:
                        response += "Sources:\n"
                        for i, doc in enumerate(result['source_documents'][:3], 1):  # Limit to top 3 sources
                            source = doc.metadata.get('source', 'Unknown source')
                            response += f"{i}. {source}\n"
                    
                    return response
                except Exception as qa_error:
                    logger.error(f"Error in QA chain: {qa_error}", exc_info=True)
                    # Fall through to simple retrieval
            
            # Fallback to simple retrieval if QA chain fails or isn't available
            logger.info("Using simple retrieval fallback")
            results = self.query(query, k=3)
            
            if not results:
                return "No relevant information found in the candidate documents."
            
            response = f"Query: {query}\n\n"
            response += "Here's what I found in the candidate documents:\n\n"
            
            for i, result in enumerate(results, 1):
                content = result.get('content', 'No content')
                source = result.get('metadata', {}).get('source', 'Unknown source')
                score = result.get('score', 1.0)
                
                response += f"{i}. [Source: {source}, Score: {score:.2f}]\n"
                response += f"   {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting candidate insights: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"An error occurred while processing your query: {error_msg}"

    def get_document_info(self) -> Dict[str, Any]:
        """
        Get information about the documents in the vector store.
        
        Returns:
            Dict[str, Any]: Document information including count and metadata
        """
        try:
            logger.info("Getting document info...")
            
            # Get document count
            count = self.get_document_count()
            
            # Initialize metadata dictionary
            metadata = {
                'document_count': count,
                'persist_directory': self.persist_directory,
                'vector_store_type': type(self.vector_store).__name__ if hasattr(self, 'vector_store') and self.vector_store else 'None'
            }
            
            # Try to get additional metadata if available
            try:
                if hasattr(self.vector_store, '_collection'):
                    collection = self.vector_store._collection
                    if hasattr(collection, 'get'):
                        result = collection.get()
                        if isinstance(result, dict):
                            metadata.update({
                                'ids_count': len(result.get('ids', [])),
                                'metadatas_count': len(result.get('metadatas', [])),
                                'embeddings_count': len(result.get('embeddings', [])) if 'embeddings' in result else 0,
                                'has_embeddings': 'embeddings' in result
                            })
                            
                            # Add sample of document sources if available
                            metadatas = result.get('metadatas', [])
                            if metadatas:
                                sources = [m.get('source', 'unknown') for m in metadatas if isinstance(m, dict)]
                                metadata['sample_sources'] = list(set(sources))[:5]  # First 5 unique sources
            except Exception as meta_error:
                logger.warning(f"Could not get detailed metadata: {meta_error}")
            
            return {
                'status': 'success',
                **metadata
            }
            
        except Exception as e:
            error_msg = f"Error getting document info: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                'status': 'error',
                'error': error_msg,
                'document_count': -1
            }

    def clear_documents(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Clearing all documents from {self.persist_directory}")
            
            # First try to use the vector store's delete method if available
            if hasattr(self.vector_store, 'delete_collection'):
                try:
                    self.vector_store.delete_collection()
                    logger.info("Successfully cleared documents using delete_collection")
                    return True
                except Exception as delete_error:
                    logger.warning(f"Could not use delete_collection: {delete_error}")
            
            # Fallback to directory deletion
            try:
                import shutil
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                
                # Recreate the directory
                os.makedirs(self.persist_directory, exist_ok=True)
                
                # Reinitialize the vector store
                self._initialize_vector_store()
                
                logger.info("Successfully cleared the vector store")
                return True
                
            except Exception as e:
                logger.error(f"Error clearing database: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error clearing documents: {e}")
            return False
            
    def clear_database(self) -> bool:
        """
        Alias for clear_documents for backward compatibility.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self.clear_documents()
