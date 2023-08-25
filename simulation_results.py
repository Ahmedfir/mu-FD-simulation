import logging
from typing import List, Set, Dict

from fd import EffortMetrics
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

    @staticmethod
    def _mean_value_for_key(d: Dict[int, float], key) -> float:
        sd = dict(sorted(d.items()))
        if key in sd:
            return sd[key]
        else:
            keys: List[int] = list(sd.keys())
            if key > keys[-1]:
                return sd[keys[-1]]
            else:
                next_k = next(k for k in keys if k > key)
                prev_k = keys[keys.index(next_k) - 1]
                assert prev_k < key < next_k
                next_v = sd[next_k]
                prev_v = sd[prev_k]
                value = prev_v + float(next_v - prev_v) * float(key - prev_k) / float(next_k - prev_k)
            return value

    def ms_mutants_at(self, effort, metric: EffortMetrics):
        if EffortMetrics.M == metric:
            return self._mean_value_for_key(self.mut_ms_dict, effort)
        elif EffortMetrics.T == metric:
            return self._mean_value_for_key(self.tests_ms_dict, effort)
        else:
            raise Exception("{0} : metric not supported!")

    @staticmethod
    def _adapt_to_new_max(d, old_max, new_max):
        return {float(effort) * float(new_max) / float(old_max): v for effort, v in d.items()}

    def adapt_to_new_max(self, old_max_mutants, new_max_mutants, old_max_tests, new_max_tests):
        assert new_max_mutants > 0
        if new_max_tests <= 0:
            logging.error('weird attempt to set max test cost to 0')
        if old_max_mutants > 0:
            self.mut_ms_dict = self._adapt_to_new_max(self.mut_ms_dict, old_max_mutants, new_max_mutants)
        else:
            logging.warning('weird simulation: old_max_mutants<=0 : no mutants generated')
        if old_max_tests > 0:
            self.tests_ms_dict = self._adapt_to_new_max(self.tests_ms_dict, old_max_tests, new_max_tests)

    def max_effort(self, metric):
        if EffortMetrics.M == metric:
            return list(sorted(self.mut_ms_dict.keys()))[-1]
        elif EffortMetrics.T == metric:
            return list(sorted(self.tests_ms_dict.keys()))[-1]
        else:
            raise Exception("{0} : metric not supported!")
