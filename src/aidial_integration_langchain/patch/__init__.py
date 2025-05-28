"""
Monkey patching langchain_openai to proxy extra parameters to the upstream model.

Workaround for https://github.com/langchain-ai/langchain/issues/26617
"""

import logging
import sys

from .decorators import (
    patch_convert_chunk_to_generation_chunk,
    patch_convert_delta_to_message_chunk,
    patch_convert_dict_to_message,
    patch_convert_message_to_dict,
    patch_create_chat_result,
)

if "langchain_openai" in sys.modules.keys():
    raise RuntimeError(
        "Import patch module before any langchain_openai imports"
    )

import langchain_openai.chat_models.base

logger = logging.getLogger(__name__)
logger.info("Patching langchain_openai library...")

# Convert OpenAI message to LC message
langchain_openai.chat_models.base._convert_message_to_dict = (
    patch_convert_message_to_dict(
        langchain_openai.chat_models.base._convert_message_to_dict
    )
)

if hasattr(langchain_openai.chat_models.base, "BaseChatOpenAI"):
    # Since langchain_openai>=0.1.5
    langchain_openai.chat_models.base.BaseChatOpenAI._create_chat_result = (
        patch_create_chat_result(
            langchain_openai.chat_models.base.BaseChatOpenAI._create_chat_result
        )
    )
elif hasattr(langchain_openai.chat_models.base, "ChatOpenAI"):
    langchain_openai.chat_models.base.ChatOpenAI._create_chat_result = (
        patch_create_chat_result(
            langchain_openai.chat_models.base.ChatOpenAI._create_chat_result
        )
    )

# Convert LC block response to OpenAI response
langchain_openai.chat_models.base._convert_dict_to_message = (
    patch_convert_dict_to_message(
        langchain_openai.chat_models.base._convert_dict_to_message
    )
)

# Convert LC streaming chunk to OpenAI streaming chunk
langchain_openai.chat_models.base._convert_delta_to_message_chunk = (
    patch_convert_delta_to_message_chunk(
        langchain_openai.chat_models.base._convert_delta_to_message_chunk
    )
)

if hasattr(
    langchain_openai.chat_models.base, "_convert_chunk_to_generation_chunk"
):
    langchain_openai.chat_models.base._convert_chunk_to_generation_chunk = (  # type: ignore
        patch_convert_chunk_to_generation_chunk(with_self=False)(
            langchain_openai.chat_models.base._convert_chunk_to_generation_chunk,  # type: ignore
        )
    )
elif hasattr(langchain_openai.chat_models.base, "BaseChatOpenAI") and hasattr(
    langchain_openai.chat_models.base.BaseChatOpenAI,
    "_convert_chunk_to_generation_chunk",
):
    # `_convert_chunk_to_generation_chunk` was moved to BaseChatOpenAI since langchain-openai==0.3.5
    langchain_openai.chat_models.base.BaseChatOpenAI._convert_chunk_to_generation_chunk = (  # type: ignore
        patch_convert_chunk_to_generation_chunk(with_self=True)(
            langchain_openai.chat_models.base.BaseChatOpenAI._convert_chunk_to_generation_chunk,  # type: ignore
        )
    )
