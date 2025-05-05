import json, os, re, requests, openai
from functools import lru_cache
from typing import Dict, List, Generator, Any, Union

from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS


OPENAI_BASE_URL      = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
MCP_SERVER_ENDPOINTS = os.getenv("MCP_SERVERS",  "http://127.0.0.1:8011").split(",")
DEBUG_OUTPUT         = bool(int(os.getenv("DEBUG", "1")))

PROXY_HOST = '0.0.0.0'
PROXY_PORT = 1235

app = Flask(__name__)
CORS(app)

def strip_thoughts(text: str | None) -> str | None:
    if not isinstance(text, str):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

def extract_api_key() -> str | None:
    return (
        request.headers.get("Api-Key")
        or (request.headers.get("Authorization") or "")
           .removeprefix("Bearer ").strip()
        or None
    )

@lru_cache
def aggregate_tools(endpoints: tuple[str, ...]) -> List[Dict]:
    combined: List[Dict] = []
    for base in endpoints:
        try:
            spec = requests.get(f"{base}/openapi.json", timeout=5).json()
        except Exception:
            continue

        for _, methods in spec.get("paths", {}).items():
            post_info = methods.get("post") or {}
            op_id = post_info.get("operationId")
            if not op_id:
                continue

            schema = (
                post_info.get("requestBody", {})
                         .get("content", {})
                         .get("application/json", {})
                         .get("schema", {})
            )
            if "$ref" in schema:
                ref_name = schema["$ref"].split("/")[-1]
                schema = spec.get("components", {}).get("schemas", {}).get(ref_name, {})
            params = schema.get("properties", {})

            combined.append(
                {
                    "type": "function",
                    "function": {
                        "name": op_id,
                        "description": post_info.get("summary", ""),
                        "parameters": {"type": "object", "properties": params or {}},
                    },
                    "mcp_server": base,
                }
            )

    dedup: Dict[str, Dict] = {}
    for t in combined:
        dedup.setdefault(t["function"]["name"], t)
    return list(dedup.values())

TOOLS = aggregate_tools(tuple(MCP_SERVER_ENDPOINTS))

def map_tool_name_to_url(name: str) -> str | None:
    for t in TOOLS:
        if t["function"]["name"] == name:
            endpoint = re.sub(r"^tool_(.*?)_post$", r"\1", name)
            return f"{t['mcp_server']}/{endpoint}"
    return None

def openai_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)


def call_openai(
    client:   openai.OpenAI,
    model:    str,
    messages: List[Dict],
    stream:   bool,
) -> Union[Generator[bytes, None, None], Dict]:
    if stream:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            tools=TOOLS, tool_choice="auto", stream=True,
        )

        def sse() -> Generator[bytes, None, None]:
            buf = ""
            t_calls: Dict[str, Dict] = {}
            last_id = None

            for chunk in resp:
                delta = chunk.choices[0].delta

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        cid = tc.id or last_id or f"tmp_{len(t_calls)}"
                        last_id = cid
                        call = t_calls.setdefault(cid, {"name": "", "arguments": ""})
                        if tc.function.name:
                            call["name"] = tc.function.name
                        if tc.function.arguments:
                            call["arguments"] += tc.function.arguments
                    continue

                if delta.content:
                    yield f"data: {json.dumps(json.loads(chunk.to_json()))}\n\n".encode()
                    buf += delta.content

            if t_calls:
                yield from run_tool_calls(t_calls, messages, client, model, stream=True)
            else:
                messages.append({"role": "assistant", "content": buf})

        return sse()

    resp = client.chat.completions.create(
        model=model, messages=messages,
        tools=TOOLS, tool_choice="auto", stream=False,
    )

    raw: Dict = resp if isinstance(resp, dict) else resp.model_dump()

    try:
        choice0 = raw["choices"][0]
        if isinstance(choice0, dict) and choice0.get("finish_reason") == "tool_calls":
            t_calls = {
                tc["id"]: {
                    "name":  tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                for tc in choice0["message"].get("tool_calls", [])
            }
            return run_tool_calls(t_calls, messages, client, model, stream=False)
    except Exception:
        pass

    return raw


def run_tool_calls(
    tool_calls: Dict[str, Dict],
    messages:   List[Dict],
    client:     openai.OpenAI,
    model:      str,
    stream:     bool,
) -> Union[Generator[bytes, None, None], Dict]:
    for cid, call in tool_calls.items():
        try:
            args = json.loads(call["arguments"] or "{}")
            url  = map_tool_name_to_url(call["name"]) or ""
            result = requests.post(url, json=args, timeout=360).json()
        except Exception as exc:
            result = {"error": str(exc)}

        messages.extend([
            {"role": "assistant",
             "tool_calls": [{
                 "id": cid, "type": "function",
                 "function": {"name": call["name"],
                              "arguments": call["arguments"]},
             }]},
            {"role": "tool", "tool_call_id": cid,
             "content": json.dumps(result, ensure_ascii=False)},
        ])

    return call_openai(client, model, messages, stream)

@app.route("/v1/chat/completions", methods=["POST"])
def completions():
    data      = request.get_json(force=True)
    stream    = bool(data.get("stream", True))
    model     = data.get("model", "")
    messages  = data.get("messages", [])
    api_key   = extract_api_key() or ""

    client = openai_client(api_key)

    if stream:
        gen = call_openai(client, model, messages, stream=True)
        return Response(stream_with_context(gen),
                        content_type="text/event-stream")

    response = call_openai(client, model, messages, stream=False)

    try:
        ch0 = response.get("choices", [])[0]
        if isinstance(ch0, dict):
            msg = ch0.get("message", {})
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = strip_thoughts(msg["content"])
    except Exception:
        pass

    return jsonify(response)

@app.route("/v1/models",  methods=["GET"])
@app.route("/v1/embeddings",  methods=["POST"])
@app.route("/v1/completions", methods=["POST"])
def forward_to_openai():
    base = OPENAI_BASE_URL.rstrip("/")
    endpoint = request.path.rsplit("/", 1)[-1]
    target_url = f"{base}/{endpoint}"
    hop = {"host", "content-length", "transfer-encoding", "connection",
           "keep-alive", "proxy-authenticate", "proxy-authorization", "upgrade"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in hop}
    resp = requests.request(
        method=request.method,
        url=target_url,
        params=request.args,
        headers=fwd_headers,
        data=request.get_data() or None,
        stream=True
    )
    excl = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    back_headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excl]

    if resp.headers.get("Content-Type", "").startswith("text/event-stream"):
        return Response(
            stream_with_context(resp.iter_content(chunk_size=None)),
            status=resp.status_code,
            headers=back_headers,
            content_type=resp.headers.get("Content-Type"),
        )

    return Response(resp.content,
                    status=resp.status_code,
                    headers=back_headers,
                    content_type=resp.headers.get("Content-Type"))

if __name__ == "__main__":
    app.run(host=PROXY_HOST, port=PROXY_PORT, debug=DEBUG_OUTPUT)
