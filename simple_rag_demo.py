"""
Simple RAG (Retrieval Augmented Generation) Example with LangGraph and Gemini
Uses simple keyword matching for retrieval to avoid dependency issues.
This demonstrates the RAG concept without requiring embeddings.
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv
from collections import Counter

from langchain_google_genai import ChatGoogleGenerativeAI
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
# This simulates a local document database
KNOWLEDGE_BASE = {
    "LangGraph": """LangGraph is a library for building stateful, multi-actor applications with LLMs.
It extends LangChain by providing a graph-based approach to orchestrate multiple chains across multiple steps.
Key features include:
- Cycles and Branching: Create loops and conditional paths (unlike simple linear chains)
- State Management: Shared state flows through the graph nodes
- Persistence: Save and resume application state
- Human-in-the-Loop: Add breakpoints for human approval before critical actions""",
    
    "LangChain": """LangChain is a framework for building applications with large language models.
Key components include:
- Prompt Templates: Standardize and parameterize prompts
- Chains: Combine multiple LLM calls and logic
- Agents: Allow LLMs to use tools and make decisions
- Memory: Keep track of conversation history
- LCEL: LangChain Expression Language for composing chains""",
    
    "RAG": """Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses
by providing relevant context from a knowledge base. The RAG process:
1. Embed documents into vectors using an embedding model
2. Store vectors in a vector database (like FAISS, Chroma, Pinecone)
3. Convert user queries to vectors
4. Find similar documents via vector similarity search
5. Provide retrieved documents as context to the LLM
Benefits: Reduces hallucinations, provides up-to-date information, grounds responses in facts""",
    
    "Vector Databases": """Vector databases store and query high-dimensional vectors efficiently.
Popular vector databases:
- FAISS: Fast, local, CPU/GPU support from Facebook AI Research
- ChromaDB: Easy to use, good for prototyping, open source
- Pinecone: Managed cloud service, highly scalable
- Weaviate: Open source with GraphQL interface
- Milvus: Open source, designed for trillion-scale vectors
They use algorithms like HNSW or IVF for approximate nearest neighbor (ANN) search""",
    
    "Agentic Workflows": """Agentic workflows involve AI agents that can autonomously perform complex tasks.
Key capabilities:
- Planning: Decompose complex tasks into subtasks
- Tool Use: Execute code, search web, access files, call APIs
- Iteration: Self-correct based on feedback and validation
- Multi-Agent: Collaborate with other specialized agents
- Human-in-Loop: Request human input for critical decisions
This represents a shift from simple prompt-response to autonomous problem solving"""
}


# Step 3: Initialize LLM
print("=" * 80)
print("🚀 Simple RAG System Initialization")
print("=" * 80)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
print("✅ Gemini LLM initialized")
print(f"📚 Knowledge base loaded with {len(KNOWLEDGE_BASE)} documents\n")


# Step 4: Simple Retrieval Function
def simple_retrieval(question: str, top_k: int = 2) -> List[tuple]:
    """
    Simple keyword-based retrieval (simulates vector similarity search)
    In production, this would be replaced with actual vector similarity search
    """
    # Tokenize question into important words (remove common words)
    stop_words = {'tell', 'me', 'about', 'what', 'is', 'how', 'does', 'the', 'a', 'an', 'and', 'or'}
    question_words = set(word.lower() for word in question.split() if word.lower() not in stop_words)
    
    # Score each document by keyword overlap
    scores = []
    for doc_title, doc_content in KNOWLEDGE_BASE.items():
        doc_text = (doc_title + " " + doc_content).lower()
        doc_words = set(doc_text.split())
        
        # Count common words between question and document
        overlap = len(question_words & doc_words)
        
        # Boost score if document title is mentioned in question
        title_match = 20 if doc_title.lower() in question.lower() else 0
        
        # Boost score if key question words appear in content (even if not exact match)
        content_match = sum(5 for word in question_words if word in doc_text)
        
        total_score = overlap + title_match + content_match
        scores.append((doc_title, doc_content, total_score))
    
    # Sort by score and return top_k
    scores.sort(key=lambda x: x[2], reverse=True)
    return [(title, content) for title, content, score in scores[:top_k]]


# Step 5: Define Node Functions

def retrieve_context(state: RAGState) -> RAGState:
    """
    Node 1: Retrieve relevant documents from knowledge base
    
    This simulates what a vector database does:
    1. Take the user's question
    2. Find most relevant documents (here using keyword matching)
    3. Return the top matches
    """
    question = state["question"]
    iteration = state.get("iteration", 0) + 1
    
    print(f"\n{'='*80}")
    print(f"🔍 STEP 1: Retrieving Relevant Context")
    print(f"{'='*80}")
    print(f"📝 Question: {question}")
    
    # Retrieve top 2 relevant documents
    retrieved = simple_retrieval(question, top_k=2)
    
    print(f"✅ Retrieved {len(retrieved)} relevant documents:")
    retrieved_texts = []
    for i, (title, content) in enumerate(retrieved, 1):
        print(f"   📄 Document {i}: {title}")
        retrieved_texts.append(f"[{title}]\n{content}")
    
    # Format context for LLM
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
    
    This is the "Generation" part of RAG:
    1. Take retrieved documents as context
    2. Create a prompt that includes both question and context
    3. Send to LLM (Gemini)
    4. Return the grounded answer
    """
    question = state["question"]
    context = state["context"]
    
    print(f"\n{'='*80}")
    print(f"🤖 STEP 2: Generating Answer with Context")
    print(f"{'='*80}")
    
    # Create a RAG prompt
    prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context documents.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the information from the context above
- Be specific and cite relevant details
- If context doesn't fully answer the question, acknowledge what's missing
- Keep your answer clear and concise

YOUR ANSWER:"""
    
    print("🔄 Sending request to Gemini...")
    
    # Make the LLM API call
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


# Step 6: Build the RAG Graph
def create_rag_graph():
    """
    Create a LangGraph workflow for RAG
    
    Graph Flow:
    START → Retrieve Context → Generate Answer → END
    
    This is a simple linear pipeline, but LangGraph allows you to add:
    - Conditional routing (if confidence is low, retrieve more)
    - Loops (regenerate if answer is poor quality)
    - Multiple nodes (fact-checking, citation extraction, etc.)
    """
    workflow = StateGraph(RAGState)
    
    # Add nodes (the processing steps)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    
    # Define the flow (the connections between nodes)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# Step 7: Main Execution
def main():
    print("\n" + "=" * 80)
    print("🧠 RAG DEMO: Retrieval Augmented Generation")
    print("   LangGraph + Gemini + Simple Keyword Retrieval")
    print("=" * 80)
    
    # Create the RAG graph
    graph = create_rag_graph()
    
    # Test questions
    questions = [
        "What is LangGraph and what makes it different from regular chains?",
        "How does RAG work and why is it useful?",
        "Tell me about FAISS and what it's used for",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'#'*80}")
        print(f"# QUESTION {i}/{len(questions)}")
        print(f"{'#'*80}")
        
        # Initial state
        initial_state = {
            "question": question,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "iteration": 0
        }
        
        # Execute the graph
        # The graph will run: retrieve → generate → end
        result = graph.invoke(initial_state)
        
        # Display final results
        print(f"\n{'='*80}")
        print(f"📊 FINAL RESULT")
        print(f"{'='*80}")
        print(f"\n❓ Question:")
        print(f"   {result['question']}")
        print(f"\n📚 Retrieved Documents:")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            title = doc.split('\n')[0]
            print(f"   {i}. {title}")
        print(f"\n💡 Generated Answer:")
        print(f"   {result['answer']}")
        print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

