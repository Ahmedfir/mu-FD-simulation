from typing import List, Set

from mutant import Mutant


class MutationScore:
    def __init__(self, killable_mutants: List[Mutant], all_mutants_l: int, ts: Set[str]):
        self.ts_size = len(ts)
        self.killable_mutants = [m for m in killable_mutants if any(t in m.failing_tests for t in ts)]
        self.all_mutants_l = all_mutants_l
        self.not_killable_mutants_length = self.all_mutants_l - len(self.killable_mutants)
        self.ms = 0.0
        self.mut_ms_dict = {0: 0.0}  # mutants_analysed: ms
        self.tests_ms_dict = {0: 0.0}  # tests written: ms

    def _recalculate_ms(self):
        surviving = len(self.killable_mutants) + self.not_killable_mutants_length
        killed = self.all_mutants_l - surviving
        self.ms = 100.0 * float(killed) / float(self.all_mutants_l)

    def update(self, mutants_analysed, tests_written, test) -> bool:
        if test is not None and len(self.killable_mutants) > 0:
            self.killable_mutants = [m for m in self.killable_mutants if test not in m.failing_tests]
            self._recalculate_ms()
        self.mut_ms_dict[mutants_analysed] = self.ms
        self.tests_ms_dict[tests_written] = self.ms
        return len(self.killable_mutants) > 0

    def set_max(self, mutants_analysed):
        self.mut_ms_dict[mutants_analysed] = self.ms
        self.tests_ms_dict[self.ts_size] = self.ms
