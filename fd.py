import logging
from enum import Enum
from os import makedirs
from os.path import join, isdir, isfile
from typing import Dict
from typing import List

from commons.pickle_utils import load_zipped_pickle, save_zipped_pickle


# todo refactor: most of code is duplicated considering mutants or tests as cost

class TargetCost:
    def __init__(self):
        self.achieved = False
        self.mutants_analysed: float = -1.0
        self.tests_written: float = -1.0

    def adapt_to_new_max(self, old_max_mutants, new_max_mutants, old_max_tests, new_max_tests):
        if self.mutants_analysed > 0.0:
            self.mutants_analysed = float(self.mutants_analysed) * float(new_max_mutants) / float(old_max_mutants)
        if self.tests_written > 0.0:
            self.tests_written = float(self.tests_written) * float(new_max_tests) / float(old_max_tests)

    def set(self, mutants_analysed, tests_written):
        self.achieved = True
        self.mutants_analysed = mutants_analysed
        self.tests_written = tests_written

    def _fd_at(self, effort, finding_effort) -> float:
        if self.achieved:
            if effort >= finding_effort:
                return 1.0
            else:
                return float(effort) / float(finding_effort)
        else:
            return 0.0

    def fd_mutants_at(self, effort):
        return self._fd_at(effort, self.mutants_analysed)

    def fd_tests_at(self, effort):
        return self._fd_at(effort, self.tests_written)


class SimulationResults:
    def __init__(self):
        self.bug_finding_cost: TargetCost = TargetCost()
        self.kill_all_mutants_cost: TargetCost = TargetCost()
        self.analyse_all_mutants: TargetCost = TargetCost()
        self.ts_evolution: Dict[int, str] = dict()

    def adapt_to_new_analyse_all_mutants_max(self, new_max_mutants: float, new_max_tests: float):
        old_max_mutants = self.analyse_all_mutants.mutants_analysed
        old_max_tests = self.analyse_all_mutants.tests_written
        if old_max_mutants < old_max_tests:
            raise Exception(
                'Wrong cost to analyse all mutants less than tests written: {0} < {1}'.format(str(old_max_mutants),
                                                                                              str(old_max_tests)))
        if old_max_mutants <= 0:
            msg = 'Wrong cost to analyse all mutants: {0}'.format(str(old_max_mutants))
            if self.bug_finding_cost.achieved:
                msg = 'No mutants analysed but bug found !? ' + msg
                raise Exception(msg)
            else:
                raise Exception(msg)

        self.bug_finding_cost.adapt_to_new_max(old_max_mutants, new_max_mutants, old_max_tests, new_max_tests)
        self.kill_all_mutants_cost.adapt_to_new_max(old_max_mutants, new_max_mutants, old_max_tests, new_max_tests)
        self.analyse_all_mutants.adapt_to_new_max(old_max_mutants, new_max_mutants, old_max_tests, new_max_tests)
        assert abs(
            self.analyse_all_mutants.mutants_analysed - new_max_mutants) < 0.00000001, "{0} instead of {1}".format(
            str(self.analyse_all_mutants.mutants_analysed), str(new_max_mutants))
        assert abs(
            self.analyse_all_mutants.tests_written - new_max_tests) < 0.00000001, "{0} instead of {1}".format(
            str(self.analyse_all_mutants.tests_written), str(new_max_tests))
        self.analyse_all_mutants.mutants_analysed = new_max_mutants
        self.analyse_all_mutants.tests_written = new_max_tests

    def on_bug_found(self, mutants_analysed, tests_written):
        self.bug_finding_cost.set(mutants_analysed, tests_written)

    def on_all_mutants_killed(self, mutants_analysed, tests_written):
        self.kill_all_mutants_cost.set(mutants_analysed, tests_written)

    def on_analysed_all_mutants(self, mutants_analysed, tests_written):
        self.analyse_all_mutants.set(mutants_analysed, tests_written)
        if int(mutants_analysed) not in self.ts_evolution.keys():
            self.ts_evolution[int(mutants_analysed)] = ''
        assert len({t for t in self.ts_evolution.values() if len(t) > 0}) == tests_written

    def is_bug_found(self):
        return self.bug_finding_cost.achieved

    def killed_all_mutants(self):
        return self.kill_all_mutants_cost.achieved

    def fd_mutants_at(self, effort) -> float:
        return self.bug_finding_cost.fd_mutants_at(effort)

    def fd_tests_at(self, effort):
        return self.bug_finding_cost.fd_tests_at(effort)


class EffortMetrics(Enum):
    M = "mutants_analaysed"
    T = "tests_written"


class SimulationsArray:
    def __init__(self, p_name, t_name, simulations, adapt_to_mean_max=True):
        self.simulations: List[SimulationResults] = [s for s in simulations if isinstance(s, SimulationResults)]
        if len(self.simulations) == 0:
            self.simulations = [si for s in simulations for si in s if isinstance(s, List)]
        if adapt_to_mean_max:
            try:
                self._stretch()
            except Exception as e:
                e.__setattr__('project', p_name)
                e.__setattr__('tool', t_name)
                raise e

    def _stretch(self):
        global_max_mutants = self.mean_max_cost(EffortMetrics.M)
        global_max_tests = self.mean_max_cost(EffortMetrics.T)
        for simulation in self.simulations:
            simulation.adapt_to_new_analyse_all_mutants_max(global_max_mutants, global_max_tests)

    @staticmethod
    def _avg(collection) -> float:
        return float(sum(collection)) / float(len(collection))

    def mean_max_cost(self, metric: EffortMetrics):
        if EffortMetrics.M == metric:
            return self._avg([res.analyse_all_mutants.mutants_analysed for res in self.simulations])
        elif EffortMetrics.T == metric:
            return self._avg([res.analyse_all_mutants.tests_written for res in self.simulations])
        else:
            raise Exception("{0} : metric not supported!")

    def max_cost(self, metric: EffortMetrics):
        if EffortMetrics.M == metric:
            return max([res.analyse_all_mutants.mutants_analysed for res in self.simulations])
        elif EffortMetrics.T == metric:
            return max([res.analyse_all_mutants.tests_written for res in self.simulations])
        else:
            raise Exception("{0} : metric not supported!")

    def mean_fd_at(self, effort, metric: EffortMetrics) -> float:
        if EffortMetrics.M == metric:
            return sum([sa.fd_mutants_at(effort) for sa in self.simulations]) / float(len(self.simulations))
        elif EffortMetrics.T == metric:
            return sum([sa.fd_tests_at(effort) for sa in self.simulations]) / float(len(self.simulations))
        else:
            raise Exception("{0} : metric not supported!")


class BugNormalisationResults:

    def __init__(self, bug_name: str, metric: EffortMetrics, max_cost_min_tool=True, steps=100,
                 all_tools_broke_tests=False, all_tools_generated_mutants=True):
        self.bug_name = bug_name
        self.metric = metric
        self.max_cost_min_tool = max_cost_min_tool
        self.steps = steps
        self.all_tools_broke_tests = all_tools_broke_tests
        self.all_tools_generated_mutants = all_tools_generated_mutants
        # tool: [effort:fd]
        self.result: Dict[str, Dict[float, float]] = dict()
        self.max_cost = -1

    def fd_by_cost(self, tools_results: Dict[str, SimulationsArray], intermediate_dir):

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
            pickle_file = join(pickle_dir, self.bug_name + '_' + '_'.join(tools_results.keys()) + '.pickle')
            if isfile(pickle_file):
                return load_zipped_pickle(pickle_file)

        # if nothing is cached
        all_max_costs = [sa.max_cost(self.metric) for sa in tools_results.values()]
        # the minimum effort 1 of the tools spent to kill all of its mutants.
        min_max_cost = min(all_max_costs)
        if min_max_cost < 1:
            if self.all_tools_broke_tests or (self.all_tools_generated_mutants and self.metric == EffortMetrics.M):
                logging.warning('{0} : one of the tools has no mutants {1}'.format(self.bug_name,
                                                                                   tools_results.keys()))
                return None
            else:
                all_max_costs = [c for c in all_max_costs if c > 0]
                if len(all_max_costs) == 0:
                    logging.warning('Skipping {0}: no tool spent any effort on this target.'.format(self.bug_name))
                    return None
                min_max_cost = min(all_max_costs)

        # stop when one of the tools reached its max cost. Or for every tool stop at its max.
        self.max_cost = min_max_cost if self.max_cost_min_tool else max(all_max_costs)

        # fixme implement the 100% separately.

        for effort_step in range(0, 101, int(100 / self.steps)):
            real_effort_step = self.max_cost * float(effort_step) / 100.0
            for tool in tools_results:

                if self.max_cost_min_tool \
                        or tools_results[tool].max_cost(self.metric) >= real_effort_step \
                        or tool not in self.result or len(self.result[tool]) < 2:
                    # if the max cost of a tool is lower than the smallest effort 1st step,
                    # we include it and do not stop at 0.0:
                    # len(self.result[tool]) < 2
                    # checks that it contains nothing or only the 0% effort.
                    if tool in self.result:
                        self.result[tool][effort_step] = tools_results[tool].mean_fd_at(real_effort_step, self.metric)
                    else:
                        self.result[tool] = {
                            effort_step: tools_results[tool].mean_fd_at(real_effort_step, self.metric)}

        if not self.max_cost_min_tool:
            for t in self.result:
                t_max_cost = tools_results[t].max_cost(self.metric)
                t_max_fd = tools_results[t].mean_fd_at(t_max_cost, self.metric)
                last_normalised_effort = list(self.result[t].keys())[-1]

                if t_max_fd > self.result[t][last_normalised_effort]:
                    # todo check this again:
                    #  what if the t_max_fd does not change but the t_max_cost is hiegher than last_normalised_effort
                    new_last_normalised_effort = float(min(
                        [st for st in range(int(last_normalised_effort), 101, int(100 / self.steps)) if
                         float(st) > last_normalised_effort]))
                    self.result[t][new_last_normalised_effort] = t_max_fd

        if pickle_file is not None:
            save_zipped_pickle(self, pickle_file)
        return self
