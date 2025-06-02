from tests.test_cases import run_test_openai_block, run_test_openai_stream
from tests.utils import get_openai_test_case


async def test_openai_stream():
    await run_test_openai_stream(get_openai_test_case())


async def test_openai_block():
    await run_test_openai_block(get_openai_test_case())
