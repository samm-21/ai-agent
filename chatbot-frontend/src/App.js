import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [query, setQuery] = useState(""); // User's query
  const [response, setResponse] = useState(""); // Model's response
  const [loading, setLoading] = useState(false); // Loading state
  const [loadingMessage, setLoadingMessage] = useState(""); // Changing loading message

  useEffect(() => {
    let interval;
    if (loading) {
      const messages = [
        "Thinking 🤔...",
        "Analyzing your query 🔍...",
        "Generating a response ✍️...",
        "Almost there 🚀...",
      ];
      let index = 0;

      // Cycle through messages every 1.5 seconds
      interval = setInterval(() => {
        setLoadingMessage(messages[index]);
        index = (index + 1) % messages.length; // Loop back to the first message
      }, 1500);
    } else {
      setLoadingMessage(""); // Clear the message when not loading
    }

    return () => clearInterval(interval); // Cleanup on unmount or when loading stops
  }, [loading]);

  const handleQuerySubmit = async () => {
    if (!query.trim()) {
      alert("Please enter a query.");
      return;
    }

    setLoading(true);
    setResponse(""); // Clear previous response

    try {
      // Send the query to the backend
      const res = await axios.post("http://localhost:5000/api/chat", { query });
      setResponse(res.data.answer); // Update the response
    } catch (error) {
      console.error("Error fetching response:", error);
      setResponse("Error: Unable to fetch response from the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Chatbot</h1>
        <div className="chat-container">
          <textarea
            placeholder="Type your query here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button onClick={handleQuerySubmit} disabled={loading}>
            {loading ? "Loading..." : "Send"}
          </button>
        </div>
        <div className="response-container">
          {loading ? (
            <p className="loading-message">{loadingMessage}</p> // Show changing text while loading
          ) : (
            <>
              <h3>Response:</h3>
              <p>{response || "No response yet."}</p>
            </>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
