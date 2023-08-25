from typing import Dict, List, Tuple

from fd import SimulationResults, EffortMetrics
from ms.bug_toolref_sim_results import BugToolSimResults
from ms.mutants_provider import MutantsProvider


class BugMutationScores:

    def __init__(self, bug_name: str, intermediate_dir: str):
        self.bug: str = bug_name
        self.intermediate_dir: str = intermediate_dir

    def cross_mutation_scores(self, tools_selections_sim_res: Dict[Tuple[str, str], List[SimulationResults]],
                              mutants_provider: MutantsProvider,
                              comparison_tools: List[str] = None,
                              effort_metric: EffortMetrics = EffortMetrics.M,
                              force_resim=False) -> Tuple[str, List[BugToolSimResults]]:
        assert effort_metric == EffortMetrics.M, 'current implementation is limited to the analysed mutants as effort.'

        res: List[BugToolSimResults] = []
        for tool_selection in tools_selections_sim_res:
            tool_selection_str = tool_selection[0] + '_' + tool_selection[1]
            bts_res = BugToolSimResults.from_cache(self.intermediate_dir, self.bug, tool_selection_str)
            if bts_res is None:
                bts_res = BugToolSimResults.from_sim_res(self.bug, tool_selection_str,
                                                         tools_selections_sim_res[tool_selection])
            if comparison_tools is None:
                comparison_tools = [ts[0] for ts in tools_selections_sim_res]
            for comp_tool in comparison_tools:
                bts_res.append_mutation_score(mutants_provider, comp_tool, self.intermediate_dir,
                                              force_resim=force_resim)
            res.append(bts_res)

        return self.bug, res
