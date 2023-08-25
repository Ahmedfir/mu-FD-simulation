import logging
from os import makedirs
from os.path import join, isdir, isfile
from typing import Dict, List

from commons.pickle_utils import load_zipped_pickle, save_zipped_pickle
from fd import EffortMetrics
from ms.bug_toolref_sim_results import BugToolSimResults


class MsEntry:
    def __init__(self, ref_tool_selection, ms_tool, is_sub=False):
        self.ref_tool_selection = ref_tool_selection
        self.ms_tool = ms_tool
        self.is_sub = is_sub

    def to_str(self):
        res = 'ms({0})_ts({1})'.format(self.ms_tool, self.ref_tool_selection)
        res = 'sub_' + res if self.is_sub else res
        return res

    def __members(self):
        return self.ref_tool_selection, self.ms_tool, self.is_sub

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())


class MsBugNormalisationResults:

    def __init__(self, bug_name: str, metric: EffortMetrics, max_cost_min_tool=True, steps=100,
                 all_tools_broke_tests=False, all_tools_generated_mutants=True):
        self.bug_name = bug_name
        self.metric = metric
        self.max_cost_min_tool = max_cost_min_tool
        self.steps = steps
        self.all_tools_broke_tests = all_tools_broke_tests
        self.all_tools_generated_mutants = all_tools_generated_mutants
        # tool: [effort:ms]
        self.result: Dict[MsEntry, Dict[float, float]] = dict()
        self.max_cost = -1

    def ms_by_cost(self, tools_results: List[BugToolSimResults], intermediate_dir):

        pickle_file = None
        # if disk caching is enabled, try to reload results form cache.
        if intermediate_dir is not None:
            pickle_dir = join(intermediate_dir, self.metric.name, str(self.steps),
                              'zoomed' if self.max_cost_min_tool else 'all')
            if not isdir(pickle_dir):
                try:
                    makedirs(pickle_dir)
                except FileExistsError as e:
                    print('concurrent dir creation : {0}'.format(pickle_dir))
            pickle_file = join(pickle_dir,
                               self.bug_name + '_' + '_'.join([tr.tool for tr in tools_results]) + '.pickle')
            if isfile(pickle_file):
                return load_zipped_pickle(pickle_file)

        # if nothing is cached
        all_max_costs = [sa.max_cost(self.metric) for sa in tools_results]
        # the minimum effort 1 of the tools spent to kill all of its mutants.
        min_max_cost = min(all_max_costs)
        if min_max_cost < 1:
            if self.all_tools_broke_tests or (self.all_tools_generated_mutants and self.metric == EffortMetrics.M):
                logging.warning('{0} : one of the tools has no mutants {1}'.format(self.bug_name,
                                                                                   [tr.tool for tr in tools_results]))
                return None
            else:
                all_max_costs = [c for c in all_max_costs if c > 0]
                if len(all_max_costs) == 0:
                    logging.warning('Skipping {0}: no tool spent any effort on this target.'.format(self.bug_name))
                    return None
                min_max_cost = min(all_max_costs)

        # stop when one of the tools reached its max cost. Or for every tool stop at its max.
        self.max_cost = min_max_cost if self.max_cost_min_tool else max(all_max_costs)

        self.result = {MsEntry(tr.tool, ms_tool, is_sub): dict() for tr in tools_results for ms_tool in
                       tr.ts_evolutions[0].mses.keys() for is_sub in [True, False]}

        # fixme implement the 100% separately.

        for effort_step in range(0, 101, int(100 / self.steps)):
            real_effort_step = self.max_cost * float(effort_step) / 100.0
            for entry in self.result:
                entry_tool_results = next(tr for tr in tools_results if entry.ref_tool_selection == tr.tool)
                if self.max_cost_min_tool \
                        or entry_tool_results.max_cost(self.metric) >= real_effort_step \
                        or entry.ref_tool_selection not in self.result or len(self.result[entry]) < 2:
                    # if the max cost of a tool is lower than the smallest effort 1st step,
                    # we include it and do not stop at 0.0:
                    # len(self.result[tool]) < 2
                    # checks that it contains nothing or only the 0% effort.
                    self.result[entry][effort_step] = entry_tool_results.mean_sub_ms_at(
                        real_effort_step, entry.ms_tool, self.metric) \
                        if entry.is_sub \
                        else entry_tool_results.mean_ms_at(
                        real_effort_step, entry.ms_tool, self.metric)

        if not self.max_cost_min_tool:
            for entry in self.result:
                entry_tool_results = next(tr for tr in tools_results if entry.ref_tool_selection == tr.tool)
                t_max_cost = entry_tool_results.max_cost(self.metric)
                t_max_ms = entry_tool_results.mean_sub_ms_at(t_max_cost, entry.ms_tool, self.metric) \
                    if entry.is_sub else entry_tool_results.mean_ms_at(t_max_cost, entry.ms_tool, self.metric)

                last_normalised_effort = list(self.result[entry].keys())[-1]

                if t_max_ms > self.result[entry][last_normalised_effort]:
                    # todo check this again:
                    #  what if the t_max_ms does not change but the t_max_cost is hiegher than last_normalised_effort
                    new_last_normalised_effort = float(min(
                        [st for st in range(int(last_normalised_effort), 101, int(100 / self.steps)) if
                         float(st) > last_normalised_effort]))
                    self.result[entry][new_last_normalised_effort] = t_max_ms

        if pickle_file is not None:
            save_zipped_pickle(self, pickle_file)
        return self
