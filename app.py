from flask import Flask, request, jsonify
from flask_cors import CORS
from agentic_research import chain  # Import your LangGraph workflow

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # Invoke the LangGraph workflow
    initial_state = {"query": query, "research_data": [], "draft_answer": None, "use_groq_openai": True}
    result = chain.invoke(initial_state)

    # Return the final answer
    return jsonify({"answer": result.get("draft_answer", "No answer generated.")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)