import copy
import random
from enum import Enum
from os import makedirs
from os.path import join, isdir, isfile

from typing import Set, List, Dict

from utils.pickle_utils import load_zipped_pickle, save_zipped_pickle

DEFAULT_EXCLUDED_TESTS = ["''", "nan", ""]


# todo refactor: most of code is duplicated considering mutants or tests as cost


class Mutant:

    def __init__(self, failing_tests: Set[str], exclude=DEFAULT_EXCLUDED_TESTS):
        self.failing_tests = {t for t in failing_tests if t not in exclude}

    def killed(self) -> bool:
        return self.failing_tests is not None and len(self.failing_tests) > 0


class TargetCost:
    def __init__(self):
        self.achieved = False
        self.mutants_analysed: int = -1
        self.tests_written: int = -1

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

    def on_bug_found(self, mutants_analysed, tests_written):
        self.bug_finding_cost.set(mutants_analysed, tests_written)

    def on_all_mutants_killed(self, mutants_analysed, tests_written):
        self.kill_all_mutants_cost.set(mutants_analysed, tests_written)

    def on_analysed_all_mutants(self, mutants_analysed, tests_written):
        self.analyse_all_mutants.set(mutants_analysed, tests_written)

    def is_bug_found(self):
        return self.bug_finding_cost.achieved

    def killed_all_mutants(self):
        return self.kill_all_mutants_cost.achieved

    def fd_mutants_at(self, effort) -> float:
        return self.bug_finding_cost.fd_mutants_at(effort)

    def fd_tests_at(self, effort):
        return self.bug_finding_cost.fd_tests_at(effort)


class Practitioner:

    def __init__(self, mutants: List[Mutant], ranked: bool = False):
        self.mutants = copy.deepcopy(mutants)
        self.ranked = ranked
        self.analysed_mutants = 0
        self.written_tests = 0

    def has_mutants(self):
        return self.mutants is not None and len(self.mutants) > 0

    def has_killable_mutants(self):
        return self.has_mutants() and any(m.killed for m in self.mutants)

    def write_test_to_kill_mutant(self, mutant):
        self.written_tests = self.written_tests + 1
        test = list(mutant.failing_tests)[random.choice(range(0, len(mutant.failing_tests)))]
        self.mutants = [m for m in self.mutants if test not in m.failing_tests]
        assert mutant not in self.mutants
        return test

    def analyse_mutant(self) -> str:
        self.analysed_mutants = self.analysed_mutants + 1
        if self.ranked:
            mutant = self.mutants[0]
        else:
            mutant = self.mutants[random.choice(range(0, len(self.mutants)))]
        test = ''
        if mutant.killed():
            test = self.write_test_to_kill_mutant(mutant)
        else:
            self.mutants.remove(mutant)
        return test

    def simulate(self, target_tests) -> SimulationResults:
        assert target_tests is not None and len(target_tests) > 0
        simulation_results = SimulationResults()
        while self.has_mutants():
            written_test = self.analyse_mutant()
            if written_test is not None:
                if not simulation_results.is_bug_found() and written_test in target_tests:
                    simulation_results.on_bug_found(self.analysed_mutants, self.written_tests)
                if not simulation_results.killed_all_mutants() and not self.has_killable_mutants():
                    simulation_results.on_all_mutants_killed(self.analysed_mutants, self.written_tests)
        assert simulation_results.killed_all_mutants()
        simulation_results.on_analysed_all_mutants(self.analysed_mutants, self.written_tests)
        return simulation_results


class Simulation:

    def __init__(self, failing_tests: Set[str]):
        self.failing_tests = failing_tests

    def process_ranked_pools(self, mutant_pools: List[List[Mutant]], repeat=100):
        result = []
        for _ in range(0, repeat):
            mutants = []
            for m_pool in mutant_pools:
                random.shuffle(m_pool)
                mutants.extend(m_pool)
            result.append(self.process(mutants, True, repeat=1))
        return result

    def process(self, mutants: List[Mutant], ranked: bool = False, repeat=100):  # default is random.
        return [Practitioner(mutants, ranked).simulate(self.failing_tests) for _ in range(0, repeat)]


class EffortMetrics(Enum):
    M = "mutants_analaysed"
    T = "tests_written"


class SimulationsArray:
    def __init__(self, simulations: List[SimulationResults]):
        self.simulations = simulations

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


class BugResults:

    def __init__(self, bug_name: str, tools_results: Dict[str, SimulationsArray]):
        self.bug_name = bug_name
        self.tools_results: Dict[str, SimulationsArray] = tools_results

    def fd_by_cost(self, metric: EffortMetrics, intermediate_dir, max_cost_min_tool=True, max_steps=100) -> Dict[
        str, Dict[float, float]]:
        pickle_file = None
        # if disk caching is enabled, try to reload results form cache.
        if intermediate_dir is not None:
            pickle_dir = join(intermediate_dir, metric.name, str(max_steps), 'zoomed' if max_cost_min_tool else 'all')
            if not isdir(pickle_dir):
                makedirs(pickle_dir)
            pickle_file = join(pickle_dir, self.bug_name + '_' + '_'.join(self.tools_results.keys()) + '.pickle')
            if isfile(pickle_file):
                return load_zipped_pickle(pickle_file)

        # tool: [effort:fd]
        result: Dict[str, Dict[float, float]] = dict()
        all_max_costs = [sa.max_cost(metric) for sa in self.tools_results.values()]
        if max_cost_min_tool:
            max_cost = min(all_max_costs)
            assert max_cost > 0, '{0} : one of the tools has no mutants {1}'.format(self.bug_name,
                                                                                    self.tools_results.keys())
            steps = max_cost if max_steps > max_cost else max_steps
            for effort_step in range(0, max_cost, int(max_cost / steps)):
                normalised_effort_step = effort_step * 100.0 / max_cost
                for tool in self.tools_results:
                    if tool in result:
                        result[tool][normalised_effort_step] = self.tools_results[tool].mean_fd_at(effort_step, metric)
                    else:
                        result[tool] = {normalised_effort_step: self.tools_results[tool].mean_fd_at(effort_step, metric)}
            for t in result:
                assert max_cost in result[t].keys()
        else:
            max_cost = max(all_max_costs)
            all_max_costs.sort()
            intervals_steps = (0 if i == 0 else all_max_costs[i - 1], m, int(m / max_steps) if m > max_steps else 1 for
                               i, m in enumerate(all_max_costs))
            for i_s in intervals_steps:
                for effort_step in range(i_s[0], i_s[1], i_s[2]):
                    normalised_effort_step = effort_step * 100.0 / max_cost
                    for tool in self.tools_results:
                        if self.tools_results[tool].max_cost(metric) > i_s[0]:
                            if tool in result:
                                result[tool][normalised_effort_step] = self.tools_results[tool].mean_fd_at(effort_step, metric)
                            else:
                                result[tool] = {normalised_effort_step: self.tools_results[tool].mean_fd_at(effort_step, metric)}
        if pickle_file is not None:
            save_zipped_pickle(result, pickle_file)
        return result
