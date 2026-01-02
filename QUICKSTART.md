# 🚀 Quick Start Guide

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your Gemini API key:**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root:
     ```bash
     echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
     ```

3. **Run the example:**
   ```bash
   python simple_gemini_graph.py
   ```

## What This Script Does

The `simple_gemini_graph.py` demonstrates:

- ✅ **State Definition**: Uses a `TypedDict` to define the graph's shared memory
- ✅ **Gemini Integration**: Makes a single API call to Gemini 1.5 Flash
- ✅ **Simple Graph**: Creates a basic graph with one node and direct flow to END
- ✅ **State Updates**: Shows how state flows through the graph

## Expected Output

```
============================================================
🧠 Simple LangGraph + Gemini Demo
============================================================

🚀 Starting graph with question: Explain what LangGraph is in 2 sentences.

🤖 Asking Gemini (iteration 1)...
📝 Question: Explain what LangGraph is in 2 sentences.
✅ Answer received: ...

============================================================
📊 FINAL RESULT
============================================================
Question: Explain what LangGraph is in 2 sentences.
Answer: [Gemini's response]
Iterations: 1
============================================================
```

## Next Steps

Once this works, you can enhance it by:
- Adding multiple nodes (e.g., researcher, writer, reviewer)
- Implementing conditional edges (routing logic)
- Adding tool calling capabilities
- Implementing persistence with checkpoints

Happy coding! 🎉

