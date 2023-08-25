from os import makedirs
from os.path import join, isfile, isdir
from typing import Dict, List

from commons.pickle_utils import load_zipped_pickle, save_zipped_pickle
from fd import SimulationResults, EffortMetrics
from ms.mutants_provider import MutantsProvider
from ms.ts_evoluation import TsEvolution


def flatten_sim_results_array(bug: str, tool: str, sim_results) -> List[SimulationResults]:
    # fixme: track down why the simulations are saved in an array of arrays and fix this.
    res = sim_results
    if sim_results and isinstance(sim_results[0], list):
        res = [res[0] for res in sim_results if res]
        if len(res) != len(sim_results):
            print('!!! {2} - {3} : obtained some empty stuff: only {0}/{1} were ok.'.format(len(res), len(sim_results),
                                                                                            bug, tool))
    return res


class BugToolSimResults:

    @staticmethod
    def from_sim_res(bug: str, tool: str, sim_results: List[SimulationResults]) -> 'BugToolSimResults':
        return BugToolSimResults(bug, tool,
                                 [sim_res.ts_evolution for sim_res in
                                  flatten_sim_results_array(bug, tool, sim_results)])

    @staticmethod
    def get_intermediate_file_name(bug: str, tool: str):
        return bug + '_' + tool + '__ms.pickle'

    @staticmethod
    def from_cache(intermediate_dir: str, bug: str, tool: str) -> 'BugToolSimResults':
        inter_file = BugToolSimResults.get_intermediate_file_name(bug, tool)
        if isfile(join(intermediate_dir, inter_file)):
            return load_zipped_pickle(join(intermediate_dir, inter_file))
        return None

    def __init__(self, bug: str, tool: str, ts_evolutions_lst: List[Dict[int, str]]):
        self.bug: str = bug
        self.tool: str = tool
        self.ts_evolutions: List[TsEvolution] = [TsEvolution(tse) for tse in ts_evolutions_lst]

    def append_mutation_score(self, mutants_provider: MutantsProvider, comp_tool: str, intermediate_dir: str,
                              force_resim=False):
        t_mutants = None
        for tse in self.ts_evolutions:
            if not tse.has_results(comp_tool):
                if t_mutants is None:
                    t_mutants = mutants_provider.get_mutants(comp_tool)
                if tse.append_tool(t_mutants, force_resim=force_resim):
                    self.to_cache(intermediate_dir)

    def to_cache(self, intermediate_dir: str):
        if intermediate_dir is not None:
            if not isdir(intermediate_dir):
                makedirs(intermediate_dir)
            save_zipped_pickle(self, join(intermediate_dir, self.get_intermediate_file_name(self.bug, self.tool)))

    def adapt_to_mean_max(self):
        global_max_mutants = self._mean_max_cost(EffortMetrics.M)
        global_max_tests = self._mean_max_cost(EffortMetrics.T)
        for tse in self.ts_evolutions:
            tse.adapt_to_new_analyse_all_mutants_max(global_max_mutants, global_max_tests)

    @staticmethod
    def _avg(collection) -> float:
        return float(sum(collection)) / float(len(collection))

    def _mean_max_cost(self, metric: EffortMetrics) -> float:
        return self._avg([tse.get_analyse_all_mutants_effort(metric) for tse in self.ts_evolutions])

    def _max_cost(self, metric: EffortMetrics):
        return max([res.get_analyse_all_mutants_effort(metric) for res in self.ts_evolutions])

    def mean_ms_at(self, effort, tool, metric: EffortMetrics) -> float:
        return sum([sa.ms_mutants_at(effort, tool, metric) for sa in self.ts_evolutions]) / float(
            len(self.ts_evolutions))

    def mean_sub_ms_at(self, effort, tool, metric: EffortMetrics) -> float:
        return sum([sa.sub_ms_mutants_at(effort, tool, metric) for sa in self.ts_evolutions]) / float(
            len(self.ts_evolutions))

    def max_cost(self, metric):
        return max([res.max_effort(metric) for res in self.ts_evolutions])
