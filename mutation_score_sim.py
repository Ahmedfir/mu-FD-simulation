from os import makedirs
from os.path import join, isfile, isdir
from typing import Dict, List

from fd import SimulationResults, EffortMetrics
from mutant import Mutant
from pickle_utils import load_zipped_pickle, save_zipped_pickle
from simulation_results import MutationScore


class ToolMutants:

    def __init__(self, bug: str, tool: str, all_mutants: List[Mutant], subsuming_mutants: List[Mutant]):
        self.bug: str = bug
        self.tool: str = tool
        self.killable_mutants: List[Mutant] = [m for m in all_mutants if m.killed()]
        self.all_mutants_l: int = len(all_mutants)
        self.subsuming_mutants: List[Mutant] = subsuming_mutants


class TsEvolution:

    def __init__(self, tse: Dict[int, str]):
        self.tse: Dict[int, str] = tse
        self.mses: Dict[str, MutationScore] = dict()
        self.subsuming_mses: Dict[str, MutationScore] = dict()

    def append_tool(self, tool: ToolMutants, force_resim=False):
        self.append_mutation_score(tool.tool, tool.killable_mutants, tool.all_mutants_l, tool.subsuming_mutants,
                                   force_resim=force_resim)

    def append_mutation_score(self, tool: str, killable_mutants: List[Mutant], all_mutants_l: int,
                              subsuming_mutants: List[Mutant], force_resim=False):
        if not force_resim and tool in self.mses.keys() and self.mses[tool] is not None:
            return
        ts = {t for t in self.tse.values() if len(t) > 0}
        ms = MutationScore(killable_mutants, all_mutants_l, ts)
        subsuming_ms = MutationScore(subsuming_mutants, len(subsuming_mutants), ts)
        for i, analysed_mut in enumerate(self.tse.keys()):
            if len(self.tse[analysed_mut]) > 0:
                has_mut = ms.update(analysed_mut, i, self.tse[analysed_mut])
                has_mut = subsuming_ms.update(analysed_mut, i, self.tse[analysed_mut]) | has_mut
                if not has_mut:
                    last_effort = sorted(self.tse.keys())[-1]
                    ms.set_max(last_effort)
                    subsuming_ms.set_max(last_effort)
            else:
                assert i == len(self.tse.keys()) - 1
                ms.set_max(analysed_mut)
                subsuming_ms.set_max(analysed_mut)
        self.mses[tool] = ms
        self.subsuming_mses[tool] = subsuming_ms


class BugToolSimResults:

    @staticmethod
    def from_sim_res(bug: str, tool: str, sim_results: List[SimulationResults]) -> 'BugToolSimResults':
        return BugToolSimResults(bug, tool, [sim_res.ts_evolution for sim_res in sim_results])

    @staticmethod
    def get_intermediate_file_name(bug: str, tool: str):
        return bug + '_' + tool + '__ms.pickle'

    @staticmethod
    def from_cache(intermediate_dir: str, bug: str, tool: str) -> 'BugToolSimResults':
        inter_file = BugToolSimResults.get_intermediate_file_name(bug, tool)
        if isfile(join(intermediate_dir, inter_file)):
            return load_zipped_pickle(join(intermediate_dir, inter_file))
        return None

    def __init__(self, bug: str, tool: str, ts_evolutions: List[Dict[int, str]]):
        self.bug: str = bug
        self.tool: str = tool
        self.ts_evolutions: List[TsEvolution] = [TsEvolution(tse) for all_mut, tse in ts_evolutions]
        self.intermediate_file_name = self.get_intermediate_file_name(bug, tool)

    def to_cache(self, intermediate_dir: str):
        if intermediate_dir is not None:
            if not isdir(intermediate_dir):
                makedirs(intermediate_dir)
            save_zipped_pickle(self, join(intermediate_dir, self.intermediate_file_name))

    def append_mutation_score(self, t_mutants: ToolMutants, intermediate_dir: str, force_resim=False):
        for tse in self.ts_evolutions:
            tse.append_tool(t_mutants, force_resim=force_resim)
            self.to_cache(intermediate_dir)


class BugMutationScores:

    def __init__(self, bug_name: str, intermediate_dir, all_tools_generated_mutants=True):
        self.bug = bug_name
        self.all_tools_generated_mutants = all_tools_generated_mutants
        self.intermediate_dir = intermediate_dir

    def cross_mutation_scores(self, tools_sim_res: Dict[str, List[SimulationResults]],
                              tools_mutants: List[ToolMutants],
                              effort_metric: EffortMetrics = EffortMetrics.M, force_resim=False):
        assert effort_metric == EffortMetrics.M, 'current implementation is limited to the analysed mutants as effort.'

        for tool in tools_sim_res:
            bts_res = BugToolSimResults.from_cache(self.intermediate_dir, self.bug, tool)
            if bts_res is None:
                bts_res = BugToolSimResults.from_sim_res(self.bug, tool, tools_sim_res[tool])
            for comp_tool in tools_mutants:
                bts_res.append_mutation_score(comp_tool, self.intermediate_dir, force_resim=force_resim)
