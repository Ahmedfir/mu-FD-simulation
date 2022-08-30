import copy
import random

from typing import Set, List


class Mutant:

    def __init__(self, failing_tests: Set[str]):
        self.failing_tests = failing_tests

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


class Practitioner:

    def __init__(self, mutants: List[Mutant], ranked: bool = False):
        self.mutants = copy.deepcopy(mutants)
        killed_mutants = {m for m in mutants if m.killed()}
        all_failing_tests = {t for m in killed_mutants for t in m.failing_tests}
        self.test_killed_mutants_map = {t: {m for m in killed_mutants if t in m.failing_tests}
                                        for t in all_failing_tests}
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
        self.mutants = [m for m in self.mutants if m not in self.test_killed_mutants_map[test]]
        del self.test_killed_mutants_map[test]
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

    def __init__(self, failing_tests: Set[str], mutants, ranked: bool = False, repeat=100):
        self.failing_tests = failing_tests
        self.ranked = ranked  # default is random.
        self.mutants = mutants
        self.repeat = repeat

    def process(self):
        return [Practitioner(self.mutants, self.ranked).simulate(self.failing_tests) for _ in range(0, self.repeat)]
