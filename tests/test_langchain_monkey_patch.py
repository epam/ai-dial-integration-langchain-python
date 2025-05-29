import pytest

from tests.test_cases import (
    run_test_langchain_block,
    run_test_langchain_streaming,
)


@pytest.mark.parametrize("is_azure", [True, False])
async def test_langchain_block(is_azure):
    await run_test_langchain_block(True, is_azure)


@pytest.mark.parametrize("is_azure", [True, False])
async def test_langchain_streaming(is_azure):
    await run_test_langchain_streaming(True, is_azure)
