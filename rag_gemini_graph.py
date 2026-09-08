"""
RAG (Retrieval Augmented Generation) Example with LangGraph and Gemini
This demonstrates a complete RAG pipeline with local vector storage.
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


# Step 1: Define the State
class RAGState(TypedDict):
    """State that flows through the RAG graph"""
    question: str                    # User's question
    retrieved_docs: List[str]        # Retrieved relevant documents
    context: str                      # Formatted context for LLM
    answer: str                       # Final answer from LLM
    iteration: int                    # Track iterations


# Step 2: Create Sample Knowledge Base
# In a real application, this would be loaded from files, databases, etc.
SAMPLE_DOCUMENTS = [
    """
    LangGraph is a library for building stateful, multi-actor applications with LLMs.
    It extends LangChain by providing a graph-based approach to orchestrate multiple
    chains (or actors) across multiple steps of computation. The key features include:
    - Cycles and Branching: Unlike simple chains, you can create loops and conditional paths
    - Persistence: Save and resume application state
    - Human-in-the-Loop: Add breakpoints for human approval
    """,
    """
    LangChain is a framework designed to simplify the creation of applications using 
    large language models. Key components include:
    - Prompt Templates: Standardize and parameterize prompts
    - Chains: Combine multiple LLM calls and logic
    - Agents: Allow LLMs to use tools and make decisions
    - Memory: Keep track of conversation history
    """,
    """
    Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses
    by providing relevant context from a knowledge base. The process involves:
    1. Embedding documents into vectors
    2. Storing vectors in a vector database
    3. Converting queries to vectors
    4. Finding similar documents via vector search
    5. Providing retrieved documents as context to the LLM
    This reduces hallucinations and provides up-to-date information.
    """,
    """
    Vector databases are specialized databases designed to store and query 
    high-dimensional vectors efficiently. Popular options include:
    - FAISS: Fast, local, CPU/GPU support
    - ChromaDB: Easy to use, good for prototyping
    - Pinecone: Managed cloud service
    - Weaviate: Open source with GraphQL
    They use algorithms like HNSW or IVF for approximate nearest neighbor search.
    """,
    """
    Agentic workflows in software development involve AI agents that can:
    - Plan and decompose complex tasks
    - Use tools (code execution, web search, file access)
    - Iterate and self-correct based on feedback
    - Collaborate with other agents
    - Request human input when needed
    This paradigm shift moves from simple prompt-response to autonomous problem solving.
    """
]


# Step 3: Initialize Components
print("🚀 Initializing components...")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# Initialize embeddings model (same provider for consistency)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Create documents from sample text
documents = [
    Document(page_content=doc.strip(), metadata={"source": f"doc_{i}"})
    for i, doc in enumerate(SAMPLE_DOCUMENTS)
]

# For this simple example, we'll use the documents as-is
# In production, you'd want to split large documents into chunks
print("📚 Creating vector store...")
vectorstore = FAISS.from_documents(documents, embeddings)
print(f"✅ Vector store created with {len(documents)} documents\n")


# Step 4: Define Node Functions

def retrieve_context(state: RAGState) -> RAGState:
    """
    Node 1: Retrieve relevant documents from vector store
    This node searches the vector database for documents similar to the question
    """
    question = state["question"]
    iteration = state.get("iteration", 0) + 1
    
    print(f"\n🔍 Retrieving context (iteration {iteration})...")
    print(f"📝 Question: {question}")
    
    # Search vector store for relevant documents (top 3)
    retrieved = vectorstore.similarity_search(question, k=3)
    
    # Extract text from retrieved documents
    retrieved_texts = [doc.page_content for doc in retrieved]
    
    print(f"✅ Retrieved {len(retrieved_texts)} relevant documents")
    for i, text in enumerate(retrieved_texts, 1):
        print(f"   Doc {i}: {text[:80]}...")
    
    # Format context for the LLM
    context = "\n\n---\n\n".join(retrieved_texts)
    
    return {
        "question": question,
        "retrieved_docs": retrieved_texts,
        "context": context,
        "answer": state.get("answer", ""),
        "iteration": iteration
    }


def generate_answer(state: RAGState) -> RAGState:
    """
    Node 2: Generate answer using retrieved context
    This node creates a prompt with context and sends it to Gemini
    """
    question = state["question"]
    context = state["context"]
    
    print(f"\n🤖 Generating answer with context...")
    
    # Create a prompt that includes the retrieved context
    prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer the question using the information from the context above. If the context doesn't contain relevant information, say so."""
    
    # Make the LLM call with context
    response = llm.invoke(prompt)
    answer = response.content
    
    print(f"✅ Answer generated: {answer[:100]}...")
    
    return {
        "question": question,
        "retrieved_docs": state["retrieved_docs"],
        "context": context,
        "answer": answer,
        "iteration": state["iteration"]
    }


# Step 5: Build the RAG Graph
def create_rag_graph():
    """
    Creates a LangGraph workflow for RAG:
    User Question -> Retrieve Context -> Generate Answer -> End
    """
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    
    # Define the flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# Step 6: Main Execution
def main():
    print("=" * 80)
    print("🧠 RAG with LangGraph + Gemini + FAISS Demo")
    print("=" * 80)
    
    # Create the graph
    graph = create_rag_graph()
    
    # Test with different questions
    questions = [
        "What is LangGraph and what are its key features?",
        "How does RAG work and why is it useful?",
        "What are some popular vector databases?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}/{len(questions)}")
        print(f"{'='*80}")
        
        # Initial state
        initial_state = {
            "question": question,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "iteration": 0
        }
        
        # Execute the graph
        result = graph.invoke(initial_state)
        
        # Display results
        print("\n" + "=" * 80)
        print("📊 RESULT")
        print("=" * 80)
        print(f"Question: {result['question']}")
        print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
        print(f"\nAnswer:\n{result['answer']}")
        print("=" * 80)


if __name__ == "__main__":
    main()

