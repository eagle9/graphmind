"""
Simple LangGraph example using Gemini API
This demonstrates a basic state-driven graph with a single LLM call.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file")


# Step 1: Define the State
# This acts as the "shared memory" that flows through the graph
class GraphState(TypedDict):
    """State that gets passed between nodes in the graph"""
    question: str
    answer: str
    iteration: int


# Step 2: Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)


# Step 3: Define Node Functions
def ask_gemini(state: GraphState) -> GraphState:
    """Node that sends a question to Gemini and gets a response"""
    question = state["question"]
    iteration = state.get("iteration", 0) + 1
    
    print(f"\n🤖 Asking Gemini (iteration {iteration})...")
    print(f"📝 Question: {question}")
    
    # Make the LLM call
    response = llm.invoke(question)
    answer = response.content
    
    print(f"✅ Answer received: {answer[:100]}...")
    
    # Update state
    return {
        "question": question,
        "answer": answer,
        "iteration": iteration
    }


# Step 4: Build the Graph
def create_graph():
    """Creates and compiles the LangGraph workflow"""
    
    # Initialize the graph with our state structure
    workflow = StateGraph(GraphState)
    
    # Add nodes (workers)
    workflow.add_node("gemini", ask_gemini)
    
    # Define the flow
    workflow.set_entry_point("gemini")
    workflow.add_edge("gemini", END)
    
    # Compile the graph
    return workflow.compile()


# Step 5: Run the Graph
def main():
    """Main execution function"""
    
    print("=" * 60)
    print("🧠 Simple LangGraph + Gemini Demo")
    print("=" * 60)
    
    # Create the graph
    graph = create_graph()
    
    # Initial state
    initial_state = {
        "question": "Explain what LangGraph is in 2 sentences.",
        "answer": "",
        "iteration": 0
    }
    
    print(f"\n🚀 Starting graph with question: {initial_state['question']}")
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULT")
    print("=" * 60)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {result['iteration']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

