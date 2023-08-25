import copy
import datetime
import logging
import random
import sys
from typing import List, Dict

from fd import SimulationResults
from mutant import Mutant

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)


class BasePractitioner:
    def __init__(self, failing_test_ids: Dict[str, int], ranked: bool = False):
        self.ranked = ranked
        self.analysed_mutants = 0
        self.written_tests = 0
        self.failing_test_ids: Dict[str, int] = failing_test_ids
        self.ids_failing_test: Dict[int, str] = dict((v, k) for k, v in failing_test_ids.items())

    def mutants_count(self) -> int:
        pass

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
        test_id: int = list(mutant.failing_tests)[random.choice(range(0, len(mutant.failing_tests)))]
        self.on_test_written(mutant, test_id)
        test_str = self.ids_failing_test[test_id]
        del self.failing_test_ids[test_str]
        del self.ids_failing_test[test_id]
        return test_str

    def simulate(self, target_tests) -> SimulationResults:
        assert target_tests is not None and len(target_tests) > 0
        simulation_results = SimulationResults()
        if not self.has_killable_mutants():
            # skip the simulation
            mu_count = self.mutants_count()
            simulation_results.on_all_mutants_killed(mu_count, 0)
            simulation_results.on_analysed_all_mutants(mu_count, 0)
            simulation_results.ts_evolution[mu_count] = []
        else:
            start = datetime.datetime.now()
            while self.has_killable_mutants():
                # log.debug("progress | {2} : m {0} t {1} ".format(str(self.analysed_mutants), str(self.written_tests),
                #                                                  str(datetime.datetime.now() - start)))
                written_test: str = self.analyse_mutant()
                if written_test is not None and len(written_test) > 0:
                    if not simulation_results.is_bug_found() and written_test in target_tests:
                        simulation_results.on_bug_found(self.analysed_mutants, self.written_tests)
                    if not simulation_results.killed_all_mutants() and not self.has_killable_mutants():
                        simulation_results.on_all_mutants_killed(self.analysed_mutants, self.written_tests)
                    simulation_results.ts_evolution[self.analysed_mutants] = written_test
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

    def __init__(self, mutants: List[Mutant], failing_test_ids=None, ranked: bool = False):
        super(Practitioner, self).__init__(failing_test_ids, ranked)
        self.mutants = copy.deepcopy(mutants)

    def mutants_count(self):
        return len(self.mutants)

    def has_mutants(self):
        return self.mutants is not None and len(self.mutants) > 0

    def has_killable_mutants(self):
        return any(True for m in self.mutants if m.killed)

    def on_test_written(self, mutant, test):
        self.mutants = [m for m in self.mutants if test not in m.failing_tests]
        assert mutant not in self.mutants

    def analyse_mutant(self) -> str:
        self.analysed_mutants = self.analysed_mutants + 1
        if self.ranked:
            mutant = self.mutants[0]
        else:
            mutant = self.mutants[random.choice(range(0, len(self.mutants)))]
        test_str = ''
        if mutant.killed:
            test_str: str = self.write_test_to_kill_mutant(mutant)
        else:
            self.mutants.remove(mutant)
        return test_str
