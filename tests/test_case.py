from __future__ import annotations

from typing import Dict, Optional

from openai import BaseModel


def _is_subdict(small: dict, big: dict) -> bool:
    for key, value in small.items():
        if key not in big or big[key] != value:
            return False
    return True


class IncludeTest(BaseModel):
    include: bool
    value: dict

    @classmethod
    def create(
        cls, include: Optional[bool], value: dict
    ) -> Optional[IncludeTest]:
        if include is None:
            return None
        return cls(include=include, value=value)

    def is_valid(self, other: dict) -> bool:
        return _is_subdict(self.value, other) == self.include

    def assert_is_valid(self, other: dict, title: str) -> None:
        if _is_subdict(self.value, other) != self.include:
            pred = "includes" if self.include else "doesn't include"
            raise AssertionError(
                f"{title}: expected that {other} {pred} all of the values from {self.value}."
            )


class TestCase(BaseModel):
    __test__ = False

    request_top_level: Optional[IncludeTest] = None
    request_tool_definition: Optional[Dict[int, Optional[IncludeTest]]] = None
    request_message: Optional[Dict[int, Optional[IncludeTest]]] = None
    response_top_level: Optional[IncludeTest] = None
    response_message: Optional[IncludeTest] = None

    def request_message_extra_fields(self, idx: int) -> Dict[int, dict]:
        if not self.request_message:
            return {}
        if test := self.request_message.get(idx):
            return test.value
        return {}

    def request_tool_definition_extra_fields(self, idx: int) -> Dict[int, dict]:
        if not self.request_tool_definition:
            return {}
        if test := self.request_tool_definition.get(idx):
            return test.value
        return {}

    @property
    def request_top_level_extra_fields(self) -> dict:
        if not self.request_top_level:
            return {}
        return self.request_top_level.value

    @property
    def response_message_extra_fields(self) -> dict:
        if not self.response_message:
            return {}
        return self.response_message.value

    @property
    def response_top_level_extra_fields(self) -> dict:
        if not self.response_top_level:
            return {}
        return self.response_top_level.value
