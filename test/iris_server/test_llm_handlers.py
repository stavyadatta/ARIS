"""Smoke test for the two OpenAI-compatible LLM handlers.

The ChatGPT and Grok handlers share their vision plumbing through
_VisionRequestMixin. Iris fails over from one to the other mid-conversation,
so they must keep identical behaviour and each keep the methods the rest of
the server calls on them.
"""

import importlib
import inspect

import numpy as np

from harness import Checks, add_iris_server_to_path, stub_core_api_package

add_iris_server_to_path()
stub_core_api_package()

_OpenAIHandler = importlib.import_module("core_api.chatgpt.chatgpt")._OpenAIHandler
_GrokHandler = importlib.import_module("core_api.grok.grok")._GrokHandler
VISION_SYSTEM_PROMPT = importlib.import_module("core_api.vision_prompt").VISION_SYSTEM_PROMPT

check = Checks()
chatgpt, grok = _OpenAIHandler(), _GrokHandler()

check.section("methods the server calls still resolve")
for name in ["_encode_image", "develop_last_message", "process_image_and_text",
             "_develop_image_system_prompt", "send_text", "img_text_response"]:
    check.equal(f"chatgpt.{name}", callable(getattr(chatgpt, name, None)), True)
    check.equal(f"grok.{name}", callable(getattr(grok, name, None)), True)
for name in ["send_o1", "send_text_get_json", "get_openai_embedding"]:
    check.equal(f"chatgpt.{name} (chatgpt only)", callable(getattr(chatgpt, name, None)), True)

check.section("each handler keeps its own model")
check.equal("chatgpt vision model", chatgpt.vision_model, "gpt-4o")
check.equal("grok vision model", grok.vision_model, "grok-2-vision-1212")

check.section("both describe the scene under one prompt")
check.equal("chatgpt prompt", chatgpt._develop_image_system_prompt(), VISION_SYSTEM_PROMPT)
check.equal("grok prompt", grok._develop_image_system_prompt(), VISION_SYSTEM_PROMPT)
check.equal("same object, so it cannot drift",
            chatgpt._develop_image_system_prompt() is grok._develop_image_system_prompt(),
            True)

check.section("shared image plumbing behaves identically")
frame = np.zeros((4, 4, 3), dtype=np.uint8)
check.equal("encoding matches", chatgpt._encode_image(frame), grok._encode_image(frame))

message = {"role": "user", "content": "hi"}
shaped = chatgpt.develop_last_message(message, "B64")
check.equal("message shape matches", shaped, grok.develop_last_message(message, "B64"))
check.equal("role preserved", shaped["role"], "user")
check.equal("text preserved", shaped["content"][0]["text"], "hi")
check.equal("image inlined as a data url",
            shaped["content"][1]["image_url"]["url"], "data:image/png;base64,B64")

from_string = chatgpt.develop_last_message("plain text", "B64")
check.equal("bare string becomes a user turn", from_string["role"], "user")
check.equal("bare string text", from_string["content"][0]["text"], "plain text")

check.section("signatures callers depend on")
signature = inspect.signature(chatgpt.process_image_and_text)
check.equal("model_name defers to the handler", signature.parameters["model_name"].default, None)
check.equal("max_tokens unchanged", signature.parameters["max_tokens"].default, 1000)
check.equal("grok accepts model_name too",
            "model_name" in inspect.signature(grok.process_image_and_text).parameters, True)

check.report("LLM HANDLERS OK")
