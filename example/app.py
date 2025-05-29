from pathlib import Path

if (Path(__file__).parent / "aidial_integration_langchain").exists():
    import aidial_integration_langchain.patch  # isort:skip  # noqa: F401 # type: ignore

import json

import httpx
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr


def _report(idx: int, received: bool, message: str):
    marker = "☑" if received else "☐"
    print(f"({idx}) {marker} {message}")


class MockClient(httpx.Client):
    def send(self, request, **kwargs):
        request_dict = json.loads(request.content.decode())

        # 1. per-message request extra
        received = request_dict["messages"][0].get("custom_content") == {
            "state": "foobar"
        }
        _report(1, received, "Request - in the `messages` list")

        # 2. top-level request extra
        received = request_dict.get("custom_fields") == {
            "configuration": {"a": "b"}
        }
        _report(2, received, "Request - on the top-level")

        message = {
            "role": "assistant",
            "content": "answer",
            "custom_content": {
                "attachments": []
            },  # 3. per-message response extra
        }

        return httpx.Response(
            request=request,
            status_code=200,
            headers={"Content-Type": "application/json"},
            json={
                "choices": [{"index": 0, "message": message}],
                "statistics": {"a": "b"},  # 4. top-level response extra
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

    request_message = HumanMessage(
        content="question",
        additional_kwargs={
            "custom_content": {"state": "foobar"}
        },  # 1. per-message request extra
    )

    print("Received extra fields in:")

    output = chat_client.generate(
        messages=[[request_message]],
        extra_body={
            "custom_fields": {"configuration": {"a": "b"}}
        },  # 2. top-level request extra
    )

    generation: ChatGeneration = output.generations[0][0]  # type: ignore
    response: BaseMessage = generation.message

    # 3. per-message response extra
    received = response.additional_kwargs.get("custom_content") == {
        "attachments": []
    }
    _report(3, received, "Response - in the `message` field")

    # 4. top-level response extra
    received = response.response_metadata.get("statistics") == {"a": "b"}
    _report(4, received, "Response - on the top-level")


if __name__ == "__main__":
    main()
