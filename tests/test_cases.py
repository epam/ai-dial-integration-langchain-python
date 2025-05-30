from typing import AsyncIterator

from openai import AsyncClient
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from tests.client import TestHTTPClient
from tests.test_case import TestCase
from tests.utils import get_langchain_test_case, with_langchain

_TOP_LEVEL_ERROR = "Unexpected result for the response top-level extra field"
_MESSAGE_ERROR = "Unexpected result for the response message extra field"


async def run_test_langchain_block(monkey_patch: bool, is_azure: bool):
    with with_langchain(is_azure, monkey_patch) as lc:
        test_case = get_langchain_test_case(monkey_patch, False)
        HumanMessage, get_client = lc

        request_message = HumanMessage(
            content="question",
            additional_kwargs=test_case.request_message_extra_fields(0),
        )

        output = await get_client(test_case).agenerate(
            messages=[[request_message]],
            extra_body=test_case.request_top_level_extra_fields,
        )

        generation = output.generations[0][0]
        response = generation.message

        if test := test_case.response_top_level_extra:
            test.assert_is_valid(response.response_metadata, _TOP_LEVEL_ERROR)

        if test := test_case.response_message_extra:
            test.assert_is_valid(response.additional_kwargs, _MESSAGE_ERROR)


async def run_test_langchain_streaming(monkey_patch: bool, is_azure: bool):
    with with_langchain(is_azure, monkey_patch) as lc:
        test_case = get_langchain_test_case(monkey_patch, True)
        HumanMessage, get_client = lc

        request_message = HumanMessage(
            content="question",
            additional_kwargs=test_case.request_message_extra_fields(0),
        )

        stream = get_client(test_case).astream(
            input=[request_message],
            extra_body=test_case.request_top_level_extra_fields,
        )

        async for chunk in stream:
            if test := test_case.response_top_level_extra:
                test.assert_is_valid(chunk.response_metadata, _TOP_LEVEL_ERROR)

            if test := test_case.response_message_extra:
                test.assert_is_valid(chunk.additional_kwargs, _MESSAGE_ERROR)


async def run_test_openai_stream(test_case: TestCase):
    http_client = TestHTTPClient(test_case=test_case)
    openai_client = AsyncClient(api_key="dummy-key", http_client=http_client)

    stream: AsyncIterator[
        ChatCompletionChunk
    ] = await openai_client.chat.completions.create(
        stream=True,
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": "question",
                **test_case.request_message_extra_fields(0),
            }  # type: ignore
        ],
        extra_body=test_case.request_top_level_extra_fields,
    )  # type: ignore

    async for c in stream:
        chunk = c.model_dump()

        if test := test_case.response_top_level_extra:
            test.assert_is_valid(chunk, _TOP_LEVEL_ERROR)

        if test := test_case.response_message_extra:
            test.assert_is_valid(chunk["choices"][0]["delta"], _MESSAGE_ERROR)


async def run_test_openai_block(test_case: TestCase):
    http_client = TestHTTPClient(test_case=test_case)
    openai_client = AsyncClient(api_key="dummy-key", http_client=http_client)

    response: ChatCompletion = await openai_client.chat.completions.create(
        stream=False,
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": "question",
                **test_case.request_message_extra_fields(0),
            }  # type: ignore
        ],
        extra_body=test_case.request_top_level_extra_fields,
    )  # type: ignore

    chunk = response.model_dump()

    if test := test_case.response_top_level_extra:
        test.assert_is_valid(chunk, _TOP_LEVEL_ERROR)

    if test := test_case.response_message_extra:
        test.assert_is_valid(chunk["choices"][0]["message"], _MESSAGE_ERROR)
