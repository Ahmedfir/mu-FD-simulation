from typing import Set

DEFAULT_EXCLUDED_TESTS = ["''", "nan", ""]


class Mutant:

    def __init__(self, failing_tests: Set[str], exclude=DEFAULT_EXCLUDED_TESTS):
        self.failing_tests = {t.strip() for t in failing_tests if t.strip() not in exclude}
        self.killed = True if self.failing_tests else False

