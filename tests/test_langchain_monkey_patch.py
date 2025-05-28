import pytest

from tests.test_cases import (
    run_test_langchain_block,
    run_test_langchain_streaming,
)
from tests.utils import PatchType


@pytest.mark.parametrize("is_azure", [True, False])
async def test_langchain_block(is_azure):
    await run_test_langchain_block(PatchType.MONKEY_PATCH, is_azure)


@pytest.mark.parametrize("is_azure", [True, False])
async def test_langchain_streaming(is_azure):
    await run_test_langchain_streaming(PatchType.MONKEY_PATCH, is_azure)
