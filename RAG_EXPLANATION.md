# RAG (Retrieval Augmented Generation) Demo

## What You Just Saw

This script demonstrates **RAG** - a technique that makes LLMs smarter by giving them access to a knowledge base!

## How It Works

### The RAG Pipeline (2 Steps):

```
User Question → [1. Retrieve Context] → [2. Generate Answer] → Final Answer
```

### Step 1: Retrieve Context (`retrieve_context` node)
- Takes your question
- Searches the knowledge base for relevant documents
- Returns the most similar documents (here we use keyword matching, but in production you'd use vector similarity)

### Step 2: Generate Answer (`generate_answer` node)
- Takes the retrieved documents as context
- Creates a prompt that includes both your question AND the context
- Sends to Gemini API
- Returns an answer grounded in the retrieved facts

## Why RAG is Powerful

Without RAG:
- LLM can only use its training data (outdated, limited)
- May hallucinate or make up facts
- No access to your private documents

With RAG:
- ✅ LLM has access to up-to-date information
- ✅ Grounded in your actual documents
- ✅ Reduced hallucinations
- ✅ Can cite sources

## The Code Structure

### State Definition
```python
class RAGState(TypedDict):
    question: str           # What the user asked
    retrieved_docs: List[str]  # Documents we found
    context: str            # Formatted context for LLM
    answer: str             # Final answer
```

### Knowledge Base
Just a Python dictionary with documents. In production, this would be:
- A vector database (FAISS, Chroma, Pinecone)
- With embeddings for semantic search

### Graph Flow
```python
workflow.set_entry_point("retrieve")    # Start here
workflow.add_edge("retrieve", "generate")  # Then go here
workflow.add_edge("generate", END)      # Then end
```

## Try It Yourself

Run different questions:
```python
questions = [
    "What is LangGraph?",
    "How does RAG work?",
    "Tell me about FAISS",  # Note: LLM correctly says "not in context"!
]
```

Notice how in Question 3, the LLM correctly says it doesn't have information about FAISS in the retrieved documents. This is RAG working properly - it only answers based on the provided context!

## Next Steps

To make this production-ready:
1. Replace keyword matching with proper vector embeddings
2. Use a real vector database (FAISS, ChromaDB)
3. Add more documents to your knowledge base
4. Implement chunking for large documents
5. Add citation tracking
6. Add quality filtering for retrieved docs

## The Power of LangGraph

This simple example shows how LangGraph makes it easy to:
- Define clear steps (nodes)
- Manage state between steps
- Create reproducible, debuggable AI workflows
- Add more nodes easily (e.g., a validation node, a citation node)

