import importlib
import sys
from contextlib import contextmanager
from enum import Enum
from importlib.metadata import version
from typing import Optional, Tuple

from packaging.version import Version

from tests.client import TestHTTPClient
from tests.test_case import IncludeTest, TestCase

inc = IncludeTest.create


def create_test_case(incs: Tuple[bool, bool, bool, bool]):
    return TestCase(
        request_top_level_extra=inc(
            incs[0], {"custom_fields": {"configuration": {"a": "b"}}}
        ),
        request_message_extra={
            0: inc(incs[1], {"custom_content": {"state": "foobar"}})
        },
        response_top_level_extra=inc(incs[2], {"statistics": {"a": "b"}}),
        response_message_extra=inc(
            incs[3], {"custom_content": {"attachments": []}}
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
def with_langchain(is_azure: bool, is_custom_class: bool, monkey_patch: bool):
    patch_module = "aidial_integration_langchain.patch"

    unload_langchain()
    if monkey_patch:
        unload_module(patch_module)
        importlib.import_module(patch_module)

    langchain_core = importlib.import_module("langchain_core")
    langchain_openai = importlib.import_module(
        "aidial_integration_langchain.langchain_openai"
        if is_custom_class
        else "langchain_openai"
    )

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


class PatchType(int, Enum):
    MONKEY_PATCH = 0
    CUSTOM_CLASS = 1


@contextmanager
def get_langchain_manager(patch_mode: Optional[PatchType], is_azure: bool):
    if patch_mode is None:
        with with_langchain(is_azure, False, False) as lc:
            yield lc
    elif patch_mode == PatchType.MONKEY_PATCH:
        with with_langchain(is_azure, False, True) as lc:
            yield lc
    elif patch_mode == PatchType.CUSTOM_CLASS:
        with with_langchain(is_azure, True, False) as lc:
            yield lc
    else:
        raise RuntimeError(f"Unexpected patch mode: {patch_mode}")


def get_openai_test_case():
    return create_test_case((True, True, True, True))


def get_langchain_test_case(
    patch_mode: Optional[PatchType], stream: bool
) -> TestCase:
    if patch_mode is None:
        return create_test_case((True, False, False, False))
    else:
        if (
            patch_mode == PatchType.MONKEY_PATCH
            and Version(version("langchain_openai")) <= Version("0.1.22")
            and stream
        ):
            # There is no easy way to patch langchain_openai<=0.1.22 to make it
            # return top-level extra response fields
            return create_test_case((True, True, False, True))
        return create_test_case((True, True, True, True))
