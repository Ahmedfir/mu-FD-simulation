from typing import Set, Dict
import copy
import random

from practitioner import BasePractitioner


class PractitionerXByPool(BasePractitioner):

    def __init__(self, mutant_pools, failing_test_ids: Dict[str, int], ranked: bool = False, max_mutants_per_pool=1):
        super(PractitionerXByPool, self).__init__(failing_test_ids, ranked)
        self.mutant_pools = copy.deepcopy(mutant_pools)
        self.max_mutants_per_pool = max_mutants_per_pool
        self.selected_pool = 0
        self.mutants_generated_from_selected_pool = 0

    def mutants_count(self) -> int:
        return len([m for p in self.mutant_pools for m in p])

    def has_mutants(self) -> bool:
        return self.mutant_pools is not None and any(True for p in self.mutant_pools if len(p) > 0)

    def has_killable_mutants(self) -> bool:
        return any(True for p in self.mutant_pools for m in p if m.killed)

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
        test_str = ''
        if mutant.killed:
            test_str: str = self.write_test_to_kill_mutant(mutant)
            if test_str is None or len(test_str) < 1:
                raise Exception()
        else:
            pool.remove(mutant)
        if self.mutants_generated_from_selected_pool == self.max_mutants_per_pool:
            self.pass_to_next_pool_index()
        return test_str

