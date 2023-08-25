from typing import Dict, List

from fd import EffortMetrics
from ms.tool_mutants import ToolMutants
from mutant import Mutant
from simulation_results import MutationScore


class TsEvolution:

    def __init__(self, tse: Dict[int, str]):
        self.tse: Dict[int, str] = tse
        self.mses: Dict[str, MutationScore] = dict()
        self.subsuming_mses: Dict[str, MutationScore] = dict()

    def append_tool(self, tool: ToolMutants, force_resim=False) -> bool:
        ''' :return True if we should cache otherwise False.'''
        return self.append_mutation_score(tool.tool, tool.killable_mutants, tool.all_mutants_l, tool.subsuming_mutants,
                                          force_resim=force_resim)

    def has_results(self, tool: str) -> bool:
        return tool in self.mses.keys() and self.mses[tool] is not None

    def append_mutation_score(self, tool: str, killable_mutants: List[Mutant], all_mutants_l: int,
                              subsuming_mutants: List[Mutant], force_resim=False) -> bool:
        ''' :return True if we should cache otherwise False.'''
        if not force_resim and self.has_results(tool):
            return False
        ts = {t for t in self.tse.values() if len(t) > 0}
        ms = MutationScore(killable_mutants, all_mutants_l, ts)
        subsuming_ms = MutationScore(subsuming_mutants, len(subsuming_mutants), ts)
        for i, analysed_mut in enumerate(self.tse.keys()):
            if len(self.tse[analysed_mut]) > 0:
                has_killable_mut = ms.update(analysed_mut, i, self.tse[analysed_mut])
                has_killable_mut = subsuming_ms.update(analysed_mut, i, self.tse[analysed_mut]) | has_killable_mut
                if not has_killable_mut:
                    last_effort = sorted(self.tse.keys())[-1]
                    ms.set_max(last_effort)
                    subsuming_ms.set_max(last_effort)
                    break
            else:
                assert i == len(self.tse.keys()) - 1
                ms.set_max(analysed_mut)
                subsuming_ms.set_max(analysed_mut)
        self.mses[tool] = ms
        self.subsuming_mses[tool] = subsuming_ms
        return True

    def adapt_to_new_analyse_all_mutants_max(self, new_max_mutants: float, new_max_tests: float):
        old_max_mutants = self.get_analyse_all_mutants_effort(EffortMetrics.M)
        old_max_tests = self.get_analyse_all_mutants_effort(EffortMetrics.T)
        if old_max_mutants < old_max_tests:
            raise Exception(
                'Wrong cost to analyse all mutants less than tests written: {0} < {1}'.format(str(old_max_mutants),
                                                                                              str(old_max_tests)))
        if old_max_mutants <= 0:
            msg = 'Wrong cost to analyse all mutants: {0}'.format(str(old_max_mutants))
            raise Exception(msg)

        for tool in self.mses:
            self.mses[tool].adapt_to_new_max(old_max_mutants, new_max_mutants, old_max_tests, new_max_tests)
            self.subsuming_mses[tool].adapt_to_new_max(old_max_mutants, new_max_mutants, old_max_tests, new_max_tests)

    def get_analyse_all_mutants_effort(self, metric: EffortMetrics = EffortMetrics.M) -> int:
        if EffortMetrics.M == metric:
            return sorted(list(self.tse.keys()))[-1]
        elif EffortMetrics.T == metric:
            return len(self.tse) - 1
        else:
            raise Exception("{0} : metric not supported!".format(metric))

    def max_effort(self, metric: EffortMetrics):
        return list(self.mses.values())[0].max_effort(metric)

    def ms_mutants_at(self, effort, tool, metric: EffortMetrics):
        return self.mses[tool].ms_mutants_at(effort, metric)

    def sub_ms_mutants_at(self, effort, tool, metric: EffortMetrics):
        return self.subsuming_mses[tool].ms_mutants_at(effort, metric)
