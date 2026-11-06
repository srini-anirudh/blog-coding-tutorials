"""Step 17 and the model clients: git checkpoints, scripted, OpenAI-compatible, HF."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from helpers import make_ctx, toy_workspace

from harness.git_tools import (checkpoint, create_checkpoint, git, git_diff, git_log, git_status,
                               list_checkpoints, restore_checkpoint, revert_to_checkpoint)
from harness.messages import ModelResponse
from harness.models import ModelError, OpenAICompatModel, ScriptedModel, ScriptExhausted, TransientModelError, call
from harness.models.openai_compat import to_wire


def test_status_diff_log(tmp_path):
    ws = toy_workspace(tmp_path)
    ctx = make_ctx(ws)
    assert "initial commit" in git_log(ctx)
    assert git_diff(ctx) == "no changes"
    (ws / "README.md").write_text("changed\n")
    assert "README.md" in git_status(ctx) and "+changed" in git_diff(ctx)


def test_checkpoint_does_not_touch_head_or_index(tmp_path):
    ws = toy_workspace(tmp_path)
    head = git(ws, "rev-parse", "HEAD")
    (ws / "new.txt").write_text("draft\n")
    create_checkpoint(ws, "wip")
    assert git(ws, "rev-parse", "HEAD") == head
    assert git(ws, "diff", "--cached", "--name-only") == ""
    assert len(list_checkpoints(ws)) == 1


def test_restore_undoes_edits_and_new_files(tmp_path):
    ws = toy_workspace(tmp_path)
    (ws / "kept.txt").write_text("untracked but checkpointed\n")
    sha = create_checkpoint(ws, "before")
    (ws / "README.md").write_text("ruined\n")
    (ws / "kept.txt").write_text("ruined\n")
    (ws / "junk.txt").write_text("should vanish\n")
    (ws / "stats_lib" / "core.py").unlink()
    removed = restore_checkpoint(ws, sha)
    assert removed == ["junk.txt"] and not (ws / "junk.txt").exists()
    assert "ruined" not in (ws / "README.md").read_text()
    assert (ws / "kept.txt").read_text().startswith("untracked") and (ws / "stats_lib" / "core.py").exists()


def test_checkpoint_tools_round_trip(tmp_path):
    ws = toy_workspace(tmp_path)
    ctx = make_ctx(ws)
    cid = checkpoint(ctx, "before").split()[1]
    (ws / "README.md").write_text("x\n")
    assert "restored" in revert_to_checkpoint(ctx, cid)
    assert git_diff(ctx) == "no changes"


def test_scripted_model_parses_raises_and_runs_out():
    boom = TransientModelError("429")
    m = ScriptedModel(["text " + call("read_file", path="a"), lambda msgs: f"saw {len(msgs)}", boom])
    first = m.complete([{"role": "user", "content": "hi"}], [])
    assert first.tool_calls[0].arguments == {"path": "a"} and first.usage.prompt_tokens > 0
    assert m.complete([{"role": "user", "content": "x"}, {"role": "user", "content": "y"}], []).text == "saw 2"
    with pytest.raises(TransientModelError):
        m.complete([], [])
    with pytest.raises(ScriptExhausted):
        m.complete([], [])


class _Fake(BaseHTTPRequestHandler):
    """A tiny OpenAI-compatible server: replays queued (status, payload) pairs."""

    queue: list[tuple[int, dict]] = []
    bodies: list[dict] = []
    headers_seen: list = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        _Fake.bodies.append(json.loads(self.rfile.read(length)))
        _Fake.headers_seen.append(dict(self.headers))
        status, payload = _Fake.queue.pop(0)
        self.send_response(status)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Fake.queue, _Fake.bodies, _Fake.headers_seen = [], [], []
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_openai_client_round_trip_and_env_config(server, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", server)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HARNESS_MODEL", "tiny")
    _Fake.queue.append((200, {"choices": [{"message": {"content": "hi", "tool_calls": [
        {"id": "abc", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3}}))
    r = OpenAICompatModel().complete([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert r.text == "hi" and r.tool_calls[0].id == "abc" and r.tool_calls[0].arguments == {"path": "a.py"}
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (7, 3)
    assert _Fake.bodies[0]["model"] == "tiny" and _Fake.bodies[0]["tool_choice"] == "auto"
    assert _Fake.headers_seen[0]["Authorization"] == "Bearer sk-test"


def test_openai_client_classifies_errors(server):
    m = OpenAICompatModel(model="m", base_url=server, api_key="k")
    _Fake.queue += [(429, {"error": "slow down"}), (503, {}), (401, {"error": "nope"})]
    for _ in range(2):
        with pytest.raises(TransientModelError):
            m.complete([], [])
    with pytest.raises(ModelError) as e:
        m.complete([], [])
    assert not isinstance(e.value, TransientModelError)


def test_openai_connection_failure_is_transient():
    m = OpenAICompatModel(model="m", base_url="http://127.0.0.1:9/v1", api_key="k", timeout_s=1)
    with pytest.raises(TransientModelError):
        m.complete([], [])


def test_openai_bad_arguments_become_parse_errors():
    r = OpenAICompatModel.parse({"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "a", "function": {"name": "x", "arguments": "{oops"}}]}}]})
    assert r.tool_calls == [] and "not valid JSON" in r.parse_errors[0] and not r.final


def test_wire_format_stringifies_arguments_without_mutating():
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "x", "arguments": {"k": 1}}}]}
    wired = to_wire([msg])[0]["tool_calls"][0]["function"]["arguments"]
    assert wired == '{"k": 1}' and msg["tool_calls"][0]["function"]["arguments"] == {"k": 1}


def test_model_response_final_flag():
    assert ModelResponse(text="done").final
    assert not ModelResponse(parse_errors=["bad"]).final


@pytest.mark.network
def test_chat_template_renders_tool_schemas_and_call_format():
    """The template, not our code, teaches the model the <tool_call> format. Check that it does."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    schema = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                                "parameters": {"type": "object", "properties": {}}}}]
    prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tools=schema,
                                     add_generation_prompt=True, tokenize=False)
    assert "read_file" in prompt and "<tool_call>" in prompt and prompt.endswith("<|im_start|>assistant\n")


@pytest.mark.network
@pytest.mark.slow
def test_hf_client_returns_a_model_response():
    from harness import HFLocalModel

    model = HFLocalModel("Qwen/Qwen2.5-0.5B-Instruct", device="cpu", max_new_tokens=8)
    r = model.complete([{"role": "user", "content": "Say hello."}], [])
    assert isinstance(r, ModelResponse) and r.usage.completion_tokens <= 8 and r.usage.prompt_tokens > 0
