### 🚀 **OpenAI Proxy with MCP Server Integration**

This is a **proxy server** that integrates multiple **MCP (Model Control Platform) servers** and provides an OpenAI-compatible API interface to interact with them. It acts as a middleman, aggregating tools from various MCP endpoints and forwarding requests accordingly.

---

### 🌐 Key Features

- **Tool Aggregation**:
  Collects available functions (`operationId`) from multiple MCP servers by fetching their `openapi.json` specifications.

- **OpenAI API Compatibility**:
  Mimics the OpenAI `/v1/chat/completions` endpoint, allowing you to use tools from your MCP servers as if they were native OpenAI functions.

- **Function Call Handling**:
  Supports streaming responses and automatically executes tool calls by forwarding requests to the appropriate MCP server endpoints.

- **Request Forwarding**:
  For non-chat routes (e.g., `/v1/models`, `/v1/embeddings`), it proxies requests directly to an OpenAI-compatible base URL.

---

### 📦 Dependencies

The code uses:
- `flask`: For handling HTTP requests.
- `requests`: To fetch OpenAPI specifications from MCP servers.
- `openai`: For interacting with the OpenAI API (e.g., model inference).
- `functools.lru_cache`: To cache aggregated tools and avoid redundant processing.

---

### 🔧 Environment Variables

| Variable Name              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `OPENAI_BASE_URL`        | Base URL for the OpenAI-compatible server (default: `http://127.0.0.1:1234/v1`).     |
| `MCP_SERVER_ENDPOINTS` | Comma-separated list of MCP servers to aggregate tools from (e.g., `http://127.0.0.1:8011`). |
| `DEBUG_OUTPUT`       | Enables debug logs (default: `True`).                                            |

---

### 🧠 How It Works

1. **Tool Discovery**:
   The server fetches OpenAPI specs from all MCP endpoints, extracts functions (`operationId`), and maps them to their respective URLs.

2. **Function Execution**:
   When a tool call is detected (e.g., `tool_id` in the response), it forwards the request to the mapped MCP server using its internal API.

3. **Response Aggregation**:
   Results from tool calls are injected back into the conversation, and the final response is returned as if the LLM generated it directly.

4. **Streaming Support**:
   The proxy supports streaming responses for real-time interaction with tools.

---

### 🧪 Example Use Case

Imagine you have multiple MCP servers hosting different functions (e.g., a weather API, a database query tool). This server allows you to:
- Use them via an OpenAI-compatible interface.
- Stream results back into the LLM's response without manual intervention.

---

### ⚙️ Setup & Run

1. **Install requirements**:
   ```bash
   pip install flask requests openai
   ```

2. **Set environment variables** (e.g., in `.env`):
   ```
   OPENAI_BASE_URL=http://localhost:1234/v1
   MCP_SERVER_ENDPOINTS=http://mcp-server-1:8011,http://mcp-server-2:8011
   DEBUG_OUTPUT=1
   ```

3. **Run the server**:
   ```bash
   python app.py
   ```

---

### 📝 Notes

- This is a **simplified example** and may need adjustments for production use (e.g., error handling, security).
- It assumes that MCP servers expose their tools via OpenAPI specs with `operationId` values.
- The code uses regex to map tool names to their corresponding endpoints.

---

### 🧩 Contributing

Feel free to open issues or PRs if you'd like to enhance the functionality (e.g., support for more complex tool schemas, better caching logic)
