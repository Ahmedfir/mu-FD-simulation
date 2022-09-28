import copy
import logging
import random
from enum import Enum
from os import makedirs
from os.path import join, isdir, isfile
from typing import Set, List, Dict

from pickle_utils import load_zipped_pickle, save_zipped_pickle

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


class BasePractitioner:
    def __init__(self, ranked: bool = False):
        self.ranked = ranked
        self.analysed_mutants = 0
        self.written_tests = 0

    def has_mutants(self) -> bool:
        """Has still mutants to analyse."""
        pass

    def has_killable_mutants(self) -> bool:
        """Has still killable mutants to analyse."""
        pass

    def analyse_mutant(self) -> str:
        """implement this to pick and analyse a mutant."""
        pass

    def on_test_written(self, mutant, test):
        """implement this to remove killed mutants by the written test."""
        pass

    def write_test_to_kill_mutant(self, mutant) -> str:
        self.written_tests = self.written_tests + 1
        test = list(mutant.failing_tests)[random.choice(range(0, len(mutant.failing_tests)))]
        self.on_test_written(mutant, test)
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


class Practitioner(BasePractitioner):

    def __init__(self, mutants: List[Mutant], ranked: bool = False):
        super(Practitioner, self).__init__(ranked)
        self.mutants = copy.deepcopy(mutants)

    def has_mutants(self):
        return self.mutants is not None and len(self.mutants) > 0

    def has_killable_mutants(self):
        return self.has_mutants() and any(m.killed for m in self.mutants)

    def on_test_written(self, mutant, test):
        self.mutants = [m for m in self.mutants if test not in m.failing_tests]
        assert mutant not in self.mutants

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


class PractitionerXByPool(BasePractitioner):

    def __init__(self, mutant_pools: List[List[Mutant]], ranked: bool = False, max_mutants_per_pool=1):
        super(PractitionerXByPool, self).__init__(ranked)
        self.mutant_pools = copy.deepcopy(mutant_pools)
        self.max_mutants_per_pool = max_mutants_per_pool
        self.selected_pool = 0
        self.mutants_generated_from_selected_pool = 0

    def has_mutants(self) -> bool:
        return self.mutant_pools is not None and len(self.mutant_pools) > 0 and any(
            len(p) > 0 for p in self.mutant_pools)

    def has_killable_mutants(self) -> bool:
        return self.has_mutants() and any(m.killed for p in self.mutant_pools for m in p)

    def on_test_written(self, mutant, test):
        self.mutant_pools = [[m for m in p if test not in m.failing_tests] for p in self.mutant_pools]
        assert mutant not in [m for p in self.mutant_pools for m in p]

    def pass_to_next_pool_index(self):
        m_pools_len = len(self.mutant_pools)
        self.selected_pool = self.selected_pool + 1 if self.selected_pool + 1 < m_pools_len else 0
        self.mutants_generated_from_selected_pool = 0

    def get_next_pool(self):
        """make sure that there's still mutants else risque of infinite loop."""
        p = self.mutant_pools[self.selected_pool]
        while p is None or len(p) == 0:
            self.pass_to_next_pool_index()
            p = self.mutant_pools[self.selected_pool]
        return p

    def analyse_mutant(self) -> str:
        pool = self.get_next_pool()
        self.mutants_generated_from_selected_pool = self.mutants_generated_from_selected_pool + 1
        self.analysed_mutants = self.analysed_mutants + 1
        if self.ranked:
            mutant = pool[0]
        else:
            mutant = pool[random.choice(range(0, len(pool)))]
        test = ''
        if mutant.killed():
            test = self.write_test_to_kill_mutant(mutant)
        else:
            pool.remove(mutant)
        if self.mutants_generated_from_selected_pool == self.max_mutants_per_pool:
            self.pass_to_next_pool_index()
        return test


class Simulation:

    def __init__(self, failing_tests: Set[str]):
        self.failing_tests = failing_tests

    def process_x_mutants_by_ranked_pool(self, mutant_pools: List[List[Mutant]], ranked: bool = False, repeat=100,
                                         max_mutants_per_pool=1):
        """ranked param controls the selection of mutants inside one pool"""
        if ranked:
            assert repeat <= 1, 'No repetition needed, if the pools and their mutants are ranked.'
            return [PractitionerXByPool(mutant_pools, ranked, max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests)]
        else:
            return [PractitionerXByPool(mutant_pools, ranked, max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests) for _ in range(0, repeat)]

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
    def __init__(self, simulations):
        self.simulations = [s for s in simulations if isinstance(s, SimulationResults)]
        if len(self.simulations) == 0:
            self.simulations = [si for s in simulations for si in s if isinstance(s, List)]

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
                 all_tools_broke_tests=False):
        self.bug_name = bug_name
        self.metric = metric
        self.max_cost_min_tool = max_cost_min_tool
        self.steps = steps
        self.all_tools_broke_tests = all_tools_broke_tests
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
                makedirs(pickle_dir)
            pickle_file = join(pickle_dir, self.bug_name + '_' + '_'.join(tools_results.keys()) + '.pickle')
            if isfile(pickle_file):
                return load_zipped_pickle(pickle_file)

        all_max_costs = [sa.max_cost(self.metric) for sa in tools_results.values()]
        if self.max_cost_min_tool:
            if self.all_tools_broke_tests:
                self.max_cost = min(all_max_costs)
                assert self.max_cost > 0, '{0} : one of the tools has no mutants {1}'.format(self.bug_name,
                                                                                             tools_results.keys())
            else:
                all_max_costs = [c for c in all_max_costs if c > 0]
                if len(all_max_costs) == 0:
                    logging.warning('Skipping {0}: no tool spent any effort on this target.'.format(self.bug_name))
                    return None
                self.max_cost = min(all_max_costs)

            for effort_step in range(0, 101, int(100 / self.steps)):
                real_effort_step = self.max_cost * float(effort_step) / 100.0
                for tool in tools_results:
                    if tool in self.result:
                        self.result[tool][effort_step] = tools_results[tool].mean_fd_at(real_effort_step, self.metric)
                    else:
                        self.result[tool] = {
                            effort_step: tools_results[tool].mean_fd_at(real_effort_step, self.metric)}
        else:
            self.max_cost = max(all_max_costs)
            for effort_step in range(0, 101, int(100 / self.steps)):
                real_effort_step = self.max_cost * float(effort_step) / 100.0
                for tool in tools_results:
                    if tools_results[tool].max_cost(self.metric) >= real_effort_step:
                        if tool in self.result:
                            self.result[tool][effort_step] = tools_results[tool].mean_fd_at(real_effort_step,
                                                                                            self.metric)
                        else:
                            self.result[tool] = {
                                effort_step: tools_results[tool].mean_fd_at(real_effort_step, self.metric)}

        if pickle_file is not None:
            save_zipped_pickle(self, pickle_file)
        return self
