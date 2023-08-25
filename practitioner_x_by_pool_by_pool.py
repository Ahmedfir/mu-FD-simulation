from typing import Set, List, Dict

from mutant import Mutant
from practitioner_x_by_pool import PractitionerXByPool


class PractitionerXByPoolPoolByPool(PractitionerXByPool):

    def __init__(self, mutant_pools: List[List[List[Mutant]]], failing_test_ids: Dict[str, int], ranked: bool = False,
                 max_mutants_per_pool=1):
        super(PractitionerXByPoolPoolByPool, self).__init__(mutant_pools, failing_test_ids=failing_test_ids,
                                                            ranked=ranked,
                                                            max_mutants_per_pool=max_mutants_per_pool)
        self.super_selected_pool = 0

    def mutants_count(self) -> int:
        return len([m for sp in self.mutant_pools for p in sp for m in p])

    def super_selected_pool_has_mutants(self) -> bool:
        return self.mutant_pools \
               and self.mutant_pools[self.super_selected_pool] \
               and any(True for p in self.mutant_pools[self.super_selected_pool] if p)

    def has_mutants(self) -> bool:
        return self.mutant_pools is not None and any(True for sp in self.mutant_pools for p in sp if p)

    def has_killable_mutants(self) -> bool:
        return any(True for sp in self.mutant_pools for p in sp for m in p if m.killed)

    def on_test_written(self, mutant, test):
        self.mutant_pools = [[[m for m in p if test not in m.failing_tests] for p in sp] for sp in self.mutant_pools]
        # assert mutant not in [m for sp in self.mutant_pools for p in sp for m in p]

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

