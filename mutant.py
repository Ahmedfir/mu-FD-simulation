from typing import Set

DEFAULT_EXCLUDED_TESTS = ["''", "nan", ""]


class Mutant:

    def __init__(self, failing_tests: Set[str], exclude=DEFAULT_EXCLUDED_TESTS):
        self.failing_tests = {t.strip() for t in failing_tests if t.strip() not in exclude}

    def killed(self) -> bool:
        return self.failing_tests is not None and len(self.failing_tests) > 0
