import importlib
import sys
from contextlib import contextmanager
from importlib.metadata import version
from typing import Optional

from packaging.version import Version

from tests.client import TestHTTPClient
from tests.test_case import IncludeTest, TestCase

inc = IncludeTest.create


def create_test_case(
    request_top_level_extra: Optional[bool] = None,
    request_message_extra: Optional[bool] = None,
    response_top_level_extra: Optional[bool] = None,
    response_message_extra: Optional[bool] = None,
) -> TestCase:
    return TestCase(
        request_top_level_extra=inc(
            request_top_level_extra,
            {"custom_fields": {"configuration": {"a": "b"}}},
        ),
        request_message_extra={
            0: inc(
                request_message_extra,
                {"custom_content": {"state": "foobar"}},
            )
        },
        response_top_level_extra=inc(
            response_top_level_extra, {"statistics": {"a": "b"}}
        ),
        response_message_extra=inc(
            response_message_extra, {"custom_content": {"attachments": []}}
        ),
    )


def unload_langchain():
    for name in list(sys.modules):
        if any(s in name for s in ["langchain", "langsmith", "pydantic"]):
            del sys.modules[name]


def unload_module(module: str):
    if module in sys.modules:
        del sys.modules[module]


@contextmanager
def with_langchain(is_azure: bool, monkey_patch: bool):
    patch_module = "aidial_integration_langchain.patch"

    unload_langchain()
    if monkey_patch:
        unload_module(patch_module)
        importlib.import_module(patch_module)

    langchain_core = importlib.import_module("langchain_core")
    langchain_openai = importlib.import_module("langchain_openai")

    def get_langchain_chat_client(test_case: TestCase):
        cls, extra_kwargs = (
            (
                langchain_openai.AzureChatOpenAI,
                {"api_version": "dummy-version", "azure_endpoint": "dummy-url"},
            )
            if is_azure
            else (langchain_openai.ChatOpenAI, {})
        )
        return cls(
            api_key="dummy-key",
            http_async_client=TestHTTPClient(test_case=test_case),
            max_retries=0,
            **extra_kwargs,
        )

    yield langchain_core.messages.HumanMessage, get_langchain_chat_client

    if monkey_patch:
        unload_module(patch_module)

    unload_langchain()


def get_openai_test_case():
    return create_test_case(
        request_top_level_extra=True,
        request_message_extra=True,
        response_top_level_extra=True,
        response_message_extra=True,
    )


def get_langchain_test_case(monkey_patch: bool, stream: bool) -> TestCase:
    if not monkey_patch:
        return create_test_case(
            request_top_level_extra=True,
            request_message_extra=False,
            response_top_level_extra=False,
            response_message_extra=False,
        )
    else:
        response_top_level_extra = True
        if Version(version("langchain_openai")) <= Version("0.1.22") and stream:
            # There is no easy way to patch langchain_openai<=0.1.22 to make it
            # return top-level extra response fields
            response_top_level_extra = False

        return create_test_case(
            request_top_level_extra=True,
            request_message_extra=True,
            response_top_level_extra=response_top_level_extra,
            response_message_extra=True,
        )
