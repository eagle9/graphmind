"""
Simple RAG (Retrieval Augmented Generation) Example with LangGraph and Gemini
Uses local embeddings (sentence-transformers) to avoid API quota issues.
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    2. Storing vectors in a vector database (like FAISS)
    3. Converting queries to vectors
    4. Finding similar documents via vector similarity search
    5. Providing retrieved documents as context to the LLM
    This reduces hallucinations and provides up-to-date, factual information.
    """,
    """
    Vector databases are specialized databases designed to store and query 
    high-dimensional vectors efficiently. Popular options include:
    - FAISS: Fast, local, CPU/GPU support from Facebook AI
    - ChromaDB: Easy to use, good for prototyping
    - Pinecone: Managed cloud service
    - Weaviate: Open source with GraphQL
    They use algorithms like HNSW or IVF for approximate nearest neighbor search.
    """,
    """
    Agentic workflows in software development involve AI agents that can:
    - Plan and decompose complex tasks into subtasks
    - Use tools (code execution, web search, file access, APIs)
    - Iterate and self-correct based on feedback and validation
    - Collaborate with other agents in multi-agent systems
    - Request human input when needed for critical decisions
    This paradigm shift moves from simple prompt-response to autonomous problem solving.
    """
]


# Step 3: Initialize Components
print("=" * 80)
print("🚀 Initializing RAG System...")
print("=" * 80)

# Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
print("✅ Gemini LLM initialized")

# Initialize local embeddings model (no API calls needed!)
# This downloads a small model (~150MB) on first run
print("📥 Loading local embeddings model (first run may take a moment)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Small, fast, and accurate
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Local embeddings model loaded")

# Create documents from sample text
documents = [
    Document(page_content=doc.strip(), metadata={"source": f"doc_{i}"})
    for i, doc in enumerate(SAMPLE_DOCUMENTS)
]

# Create vector store with local embeddings
print("📚 Creating FAISS vector store...")
vectorstore = FAISS.from_documents(documents, embeddings)
print(f"✅ Vector store created with {len(documents)} documents\n")


# Step 4: Define Node Functions

def retrieve_context(state: RAGState) -> RAGState:
    """
    Node 1: Retrieve relevant documents from vector store
    
    This node:
    1. Takes the user's question
    2. Converts it to a vector using embeddings
    3. Searches FAISS for similar document vectors
    4. Returns the most relevant documents
    """
    question = state["question"]
    iteration = state.get("iteration", 0) + 1
    
    print(f"\n🔍 STEP 1: Retrieving context (iteration {iteration})")
    print(f"📝 Question: {question}")
    
    # Search vector store for top 3 most relevant documents
    # This uses cosine similarity between question embedding and document embeddings
    retrieved = vectorstore.similarity_search(question, k=3)
    
    # Extract text content from retrieved documents
    retrieved_texts = [doc.page_content for doc in retrieved]
    
    print(f"✅ Retrieved {len(retrieved_texts)} relevant documents:")
    for i, text in enumerate(retrieved_texts, 1):
        preview = text.replace('\n', ' ')[:100]
        print(f"   📄 Doc {i}: {preview}...")
    
    # Format all retrieved documents into a single context string
    context = "\n\n---DOCUMENT SEPARATOR---\n\n".join(retrieved_texts)
    
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
    
    This node:
    1. Takes the retrieved context and question
    2. Creates a prompt that includes both
    3. Sends to Gemini API
    4. Returns the generated answer
    """
    question = state["question"]
    context = state["context"]
    
    print(f"\n🤖 STEP 2: Generating answer with Gemini")
    
    # Create a RAG prompt that instructs the LLM to use the context
    prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context documents.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer the question using information from the context above
- Be specific and cite relevant details from the context
- If the context doesn't contain enough information, say so clearly
- Keep your answer concise but informative

ANSWER:"""
    
    # Make the LLM call with the RAG prompt
    response = llm.invoke(prompt)
    answer = response.content
    
    print(f"✅ Answer generated ({len(answer)} characters)")
    
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
    
    Flow: Question → Retrieve Context → Generate Answer → End
    
    This is a simple linear flow, but you could add:
    - Conditional logic (if context is insufficient, try different search)
    - Loops (re-retrieve with refined query)
    - Multiple generation nodes (generate, validate, regenerate)
    """
    workflow = StateGraph(RAGState)
    
    # Add nodes (the workers)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    
    # Define the flow (the connections)
    workflow.set_entry_point("retrieve")  # Start here
    workflow.add_edge("retrieve", "generate")  # Then go here
    workflow.add_edge("generate", END)  # Then end
    
    return workflow.compile()


# Step 6: Main Execution
def main():
    print("\n" + "=" * 80)
    print("🧠 RAG DEMO: LangGraph + Gemini + FAISS + Local Embeddings")
    print("=" * 80)
    
    # Create the graph
    graph = create_rag_graph()
    
    # Test with different questions
    questions = [
        "What is LangGraph and what are its key features?",
        "Explain how RAG works step by step.",
        "What vector databases are mentioned and which one is from Facebook?",
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
        
        # Execute the graph (runs both nodes in sequence)
        result = graph.invoke(initial_state)
        
        # Display final results
        print("\n" + "=" * 80)
        print("📊 FINAL RESULT")
        print("=" * 80)
        print(f"\n❓ Question:\n{result['question']}")
        print(f"\n💾 Retrieved {len(result['retrieved_docs'])} documents from vector DB")
        print(f"\n✨ Answer:\n{result['answer']}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

