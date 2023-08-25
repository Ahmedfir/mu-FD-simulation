from typing import Set, Dict

DEFAULT_EXCLUDED_TESTS = ["''", "nan", ""]


class Mutant:

    def __init__(self, failing_tests: Set, exclude=DEFAULT_EXCLUDED_TESTS):
        self.failing_tests = {self._adapt_test_by_type(t) for t in failing_tests if
                              isinstance(t, int) or (isinstance(t, str) and t.strip() not in exclude)}
        self.killed = True if self.failing_tests else False

    def _adapt_test_by_type(self, t):
        return t.strip() if isinstance(t, str) else t

    def int_copy(self, test_ids: Dict[str, int]):
        return Mutant({test_ids[t] for t in self.failing_tests})
