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
        self.failing_tests = {t.strip() for t in failing_tests if t.strip() not in exclude}

    def killed(self) -> bool:
        return self.failing_tests is not None and len(self.failing_tests) > 0


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

    def mutants_count(self) -> int:
        pass

    def has_mutants(self) -> bool:
        """Has still mutants to analyse."""
        pass

    def has_killable_mutants(self) -> bool:
        """Has still killable mutants to analyse."""
        pass

    def get_killable_mutants_tests(self) -> Set[str]:
        """tests failing by killable mutants."""
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
        if not self.has_killable_mutants():
            # skip the simulation
            mu_count = self.mutants_count()
            simulation_results.on_all_mutants_killed(mu_count, 0)
            simulation_results.on_analysed_all_mutants(mu_count, 0)
        else:
            while self.has_killable_mutants():
                written_test = self.analyse_mutant()
                if written_test is not None and len(written_test) > 0:
                    if not simulation_results.is_bug_found() and written_test in target_tests:
                        simulation_results.on_bug_found(self.analysed_mutants, self.written_tests)
                    if not simulation_results.killed_all_mutants() and not self.has_killable_mutants():
                        simulation_results.on_all_mutants_killed(self.analysed_mutants, self.written_tests)
                    # else:
                    #     tests_killable_mutants = self.get_killable_mutants_tests()
                    #     l_tests_killable_mutants = len(tests_killable_mutants)
                    #     if l_tests_killable_mutants == 0:
                    #         raise Exception(
                    #             'no mutant has broken tests but self.has_killable_mutants() returns {0}'.format(
                    #                 str(self.has_killable_mutants())))

            assert not self.has_killable_mutants()
            assert simulation_results.killed_all_mutants()
            if self.has_mutants():
                self.analysed_mutants = self.analysed_mutants + self.mutants_count()
            simulation_results.on_analysed_all_mutants(self.analysed_mutants, self.written_tests)
        return simulation_results


class Practitioner(BasePractitioner):

    def __init__(self, mutants: List[Mutant], ranked: bool = False):
        super(Practitioner, self).__init__(ranked)
        self.mutants = copy.deepcopy(mutants)

    def mutants_count(self):
        return len(self.mutants)

    def has_mutants(self):
        return self.mutants is not None and len(self.mutants) > 0

    def has_killable_mutants(self):
        return any(m.killed() for m in self.mutants)

    def get_killable_mutants_tests(self) -> Set[str]:
        res = set()
        if self.has_mutants():
            res = {t for m in self.mutants for t in m.failing_tests}
        return res

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

    def __init__(self, mutant_pools, ranked: bool = False, max_mutants_per_pool=1):
        super(PractitionerXByPool, self).__init__(ranked)
        self.mutant_pools = copy.deepcopy(mutant_pools)
        self.max_mutants_per_pool = max_mutants_per_pool
        self.selected_pool = 0
        self.mutants_generated_from_selected_pool = 0

    def mutants_count(self) -> int:
        return len([m for p in self.mutant_pools for m in p])

    def has_mutants(self) -> bool:
        return self.mutant_pools is not None and any(len(p) > 0 for p in self.mutant_pools)

    def has_killable_mutants(self) -> bool:
        return any(m.killed() for p in self.mutant_pools for m in p)

    def get_killable_mutants_tests(self) -> Set[str]:
        res = set()
        if self.has_mutants():
            res = {t for p in self.mutant_pools for m in p for t in m.failing_tests}
        return res

    def on_test_written(self, mutant, test):
        self.mutant_pools = [[m for m in p if test not in m.failing_tests] for p in self.mutant_pools]
        assert mutant not in [m for p in self.mutant_pools for m in p]

    def pass_to_next_pool_index(self):
        m_pools_len = len(self.mutant_pools)
        self.selected_pool = self.selected_pool + 1 if self.selected_pool + 1 < m_pools_len else 0
        self.mutants_generated_from_selected_pool = 0

    def get_next_pool(self):
        """make sure that there's still mutants else risque of crash."""
        p = next((x for x in self.mutant_pools[self.selected_pool:] if x is not None and len(x) > 0), None)
        self.mutant_pools = [x for x in self.mutant_pools if x is not None and len(x) > 0]
        if p is None:
            p = self.mutant_pools[0]
            self.mutants_generated_from_selected_pool = 0
        self.selected_pool = self.mutant_pools.index(p)
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
            if test is None or len(test) < 1:
                raise Exception()
        else:
            pool.remove(mutant)
        if self.mutants_generated_from_selected_pool == self.max_mutants_per_pool:
            self.pass_to_next_pool_index()
        return test


class PractitionerXByPoolPoolByPool(PractitionerXByPool):

    def __init__(self, mutant_pools: List[List[List[Mutant]]], ranked: bool = False, max_mutants_per_pool=1):
        super(PractitionerXByPoolPoolByPool, self).__init__(mutant_pools, ranked=ranked,
                                                            max_mutants_per_pool=max_mutants_per_pool)
        self.super_selected_pool = 0

    def mutants_count(self) -> int:
        return len([m for sp in self.mutant_pools for p in sp for m in p])

    def super_selected_pool_has_mutants(self) -> bool:
        return self.mutant_pools is not None and len(self.mutant_pools) > 0 \
               and self.mutant_pools[self.super_selected_pool] is not None \
               and len(self.mutant_pools[self.super_selected_pool]) > 0 \
               and any(len(p) > 0 for p in self.mutant_pools[self.super_selected_pool])

    def has_mutants(self) -> bool:
        return self.mutant_pools is not None and any(len(p) > 0 for sp in self.mutant_pools for p in sp)

    def has_killable_mutants(self) -> bool:
        return any(m.killed() for sp in self.mutant_pools for p in sp for m in p)

    def get_killable_mutants_tests(self) -> Set[str]:
        res = set()
        if self.has_mutants():
            res = {t for sp in self.mutant_pools for p in sp for m in p for t in m.failing_tests}
        return res

    def on_test_written(self, mutant, test):
        self.mutant_pools = [[[m for m in p if test not in m.failing_tests] for p in sp] for sp in self.mutant_pools]
        assert mutant not in [m for sp in self.mutant_pools for p in sp for m in p]

    def pass_to_next_pool_index(self):
        if not self.super_selected_pool_has_mutants():
            super_m_pools_len = len(self.mutant_pools)
            if self.super_selected_pool + 1 < super_m_pools_len:
                self.super_selected_pool = self.super_selected_pool + 1
                self.selected_pool = 0
            else:
                assert not self.has_mutants(), 'has_mutants returns True but we finished all pools.'
        else:
            m_pools_len = len(self.mutant_pools[self.super_selected_pool])
            self.selected_pool = self.selected_pool + 1 if self.selected_pool + 1 < m_pools_len else 0
        self.mutants_generated_from_selected_pool = 0

    def get_next_pool(self):
        """make sure that there's still mutants else risque of infinite loop."""
        while not (self.super_selected_pool < len(self.mutant_pools)
                   and self.mutant_pools[self.super_selected_pool] is not None
                   and self.selected_pool < len(self.mutant_pools[self.super_selected_pool])
                   and self.mutant_pools[self.super_selected_pool][self.selected_pool] is not None
                   and len(self.mutant_pools[self.super_selected_pool][self.selected_pool]) > 0):
            self.pass_to_next_pool_index()
        sp = self.mutant_pools[self.super_selected_pool]
        p = sp[self.selected_pool]
        sp = [x for x in self.mutant_pools[self.super_selected_pool] if x is not None and len(x) > 0]
        self.mutant_pools = [[x for x in super_pool if x is not None and len(x) > 0]
                             for super_pool in self.mutant_pools
                             if super_pool is not None and len(super_pool) > 0]
        self.super_selected_pool = self.mutant_pools.index(sp)
        self.selected_pool = sp.index(p)
        return p


class Simulation:

    def __init__(self, failing_tests: Set[str]):
        self.failing_tests = failing_tests

    def process_x_ranked_mutants_by_ranked_pool(self, mutant_pools: List[List[List[Mutant]]], repeat=100,
                                                max_mutants_per_pool=1):
        result = []
        for _ in range(0, repeat):
            pools = []
            for p_pool in mutant_pools:
                p = []
                for m_pool in p_pool:
                    random.shuffle(m_pool)
                    p.extend(m_pool)
                pools.append(p)
            result.append(self.process_x_mutants_by_ranked_pool(pools, ranked=True, repeat=1,
                                                                max_mutants_per_pool=max_mutants_per_pool))
        return result

    def process_x_mutants_by_ranked_pool(self, mutant_pools: List[List[Mutant]], ranked: bool = False, repeat=100,
                                         max_mutants_per_pool=1):
        """X (max_mutants_per_pool) mutants are selected from each pool before passing to the next one.
        The pools are traversed in the given order.
        Once all pools are traversed, we restart from the first one again, picking X mutants by pool, etc.
        ranked param controls the selection of mutants inside one pool."""
        if ranked:
            assert repeat <= 1, 'No repetition needed, if the pools and their mutants are ranked.'
            return [PractitionerXByPool(mutant_pools, ranked, max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests)]
        else:
            return [PractitionerXByPool(mutant_pools, ranked, max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests) for _ in range(0, repeat)]

    def process_x_mutants_by_ranked_pool_by_ranked_pools(self, mutant_pools: List[List[List[Mutant]]],
                                                         ranked: bool = False, repeat=100, max_mutants_per_pool=1):
        """the higher level of pools or super_pools (each pool contains pools of mutants) are
        ranked and traversed in that order until finished:
        we finish all mutants of the pools of the super_pool before passing to the next super_pool.
        inside the super_pool, the pools traversing and mutants selection is similar to the one of
        process_x_mutants_by_ranked_pool().
        ranked param controls the selection of mutants inside one pool."""
        if ranked:
            assert repeat <= 1, 'No repetition needed, if the pools and their mutants are ranked.'
            return [PractitionerXByPoolPoolByPool(mutant_pools, ranked=ranked,
                                                  max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests)]
        else:
            return [PractitionerXByPoolPoolByPool(mutant_pools, ranked=ranked,
                                                  max_mutants_per_pool=max_mutants_per_pool).simulate(
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
    def __init__(self, p_name, t_name, simulations, adapt_to_mean_max=True):
        self.simulations = [s for s in simulations if isinstance(s, SimulationResults)]
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
                try:
                    makedirs(pickle_dir)
                except FileExistsError as e:
                    print('concurrent dir creation : {0}'.format(pickle_dir))
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
