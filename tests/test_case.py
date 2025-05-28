from typing import Dict

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
    def create(cls, include: bool, value: dict):
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

    request_top_level_extra: IncludeTest
    request_message_extra: Dict[int, IncludeTest]
    response_top_level_extra: IncludeTest
    response_message_extra: IncludeTest
