import importlib.util
import sys

if len(sys.argv) > 1 and sys.argv[1] == "patch":
    if not importlib.util.find_spec("aidial_integration_langchain"):
        print(
            "Error: aidial-integration-langchain package isn't installed. "
            "Try running `pip install aidial-integration-langchain` first."
        )
        sys.exit(1)

    import os

    os.environ["LC_EXTRA_REQUEST_MESSAGE_FIELDS"] = "extra_field"
    os.environ["LC_EXTRA_RESPONSE_MESSAGE_FIELDS"] = "extra_field"
    os.environ["LC_EXTRA_RESPONSE_FIELDS"] = "extra_field"
    import aidial_integration_langchain.patch  # isort:skip  # noqa: F401 # type: ignore

import json

import httpx
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

_counter = 0


def _report(received: bool, message: str):
    global _counter
    _counter += 1
    marker = "☑" if received else "☐"
    print(f"({_counter}) {marker} {message}")


class MockClient(httpx.Client):
    def send(self, request, **kwargs):
        request_dict = json.loads(request.content.decode())

        if "tools" in request_dict:
            # 5. per-tool definition extra
            received = (
                request_dict["tools"][0].get("extra_field")
                == "request.tools[0].extra_field"
            )
            _report(received, "request.tools[0].extra_field")
        else:

            # 1. per-message request extra
            received = (
                request_dict["messages"][0].get("extra_field")
                == "request.messages[0].extra_field"
            )
            _report(received, "request.messages[0].extra_field")

            # 2. top-level request extra
            received = request_dict.get("extra_field") == "request.extra_field"
            _report(received, "request.extra_field")

        response_message = {
            "role": "assistant",
            "content": "answer",
            "extra_field": "response.message.extra_field",  # 3. per-message response extra
        }

        return httpx.Response(
            request=request,
            status_code=200,
            headers={"Content-Type": "application/json"},
            json={
                "choices": [{"index": 0, "message": response_message}],
                "extra_field": "response.extra_field",  # 4. top-level response extra
            },
        )


def main():

    chat_client = AzureChatOpenAI(
        api_key=SecretStr("dummy-key"),
        api_version="dummy-version",
        azure_endpoint="dummy-url",
        http_client=MockClient(),
        max_retries=0,
    )

    print("Received the following extra fields:")

    tool = {
        "type": "function",
        "function": {"name": "dummy_tool", "parameters": {}},
        "extra_field": "request.tools[0].extra_field",  # 5. per-tool definition extra
    }
    chat_client.bind_tools(tools=[tool]).invoke("2+2=?")

    request_message = HumanMessage(
        content="question",
        additional_kwargs={
            "extra_field": "request.messages[0].extra_field"
        },  # 1. per-message request extra
    )

    output = chat_client.generate(
        messages=[[request_message]],
        extra_body={
            "extra_field": "request.extra_field"
        },  # 2. top-level request extra
    )

    generation: ChatGeneration = output.generations[0][0]  # type: ignore
    response: BaseMessage = generation.message

    # 3. per-message response extra
    received = (
        response.additional_kwargs.get("extra_field")
        == "response.message.extra_field"
    )
    _report(received, "response.message.extra_field")

    # 4. top-level response extra
    received = (
        response.response_metadata.get("extra_field") == "response.extra_field"
    )
    _report(received, "response.extra_field")


if __name__ == "__main__":
    main()
