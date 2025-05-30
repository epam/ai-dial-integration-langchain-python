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

    request_top_level_extra: Optional[IncludeTest] = None
    request_message_extra: Optional[Dict[int, Optional[IncludeTest]]] = None
    response_top_level_extra: Optional[IncludeTest] = None
    response_message_extra: Optional[IncludeTest] = None

    def request_message_extra_fields(self, idx: int) -> Dict[int, dict]:
        if not self.request_message_extra:
            return {}
        if test := self.request_message_extra.get(idx):
            return test.value
        return {}

    @property
    def request_top_level_extra_fields(self) -> dict:
        if not self.request_top_level_extra:
            return {}
        return self.request_top_level_extra.value

    @property
    def response_message_extra_fields(self) -> dict:
        if not self.response_message_extra:
            return {}
        return self.response_message_extra.value

    @property
    def response_top_level_extra_fields(self) -> dict:
        if not self.response_top_level_extra:
            return {}
        return self.response_top_level_extra.value
