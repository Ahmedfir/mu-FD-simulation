import random
from os import makedirs
from os.path import join, isfile, isdir
from pathlib import Path
from typing import Set, List

from commons.pool_executors import process_parallel_run
from mutant import Mutant
from commons.pickle_utils import load_zipped_pickle
from practitioner import Practitioner
from practitioner_x_by_pool import PractitionerXByPool
from practitioner_x_by_pool_by_pool import PractitionerXByPoolPoolByPool


def load_cached_progress(cache_dir, tmp_file) -> List:
    result = []
    if cache_dir is not None and tmp_file is not None:
        pickle_file = join(cache_dir, tmp_file)
        if isfile(pickle_file):
            result = load_zipped_pickle(join(cache_dir, tmp_file))
        elif not isdir(Path(pickle_file).parent):
            try:
                makedirs(Path(pickle_file).parent)
            except FileExistsError:
                print("two threads created the directory concurrently.")
    return result


class Simulation:

    def __init__(self, target_tests: Set[str]):
        self.failing_tests: Set[str] = target_tests

    def process_x_mutants_by_ranked_pool_by_random_pools(self, mutant_pools: List[List[List[Mutant]]],
                                                         tmp_file, failing_test_ids=None,
                                                         cache_dir='process_x_mutants_by_ranked_pool_by_random_pools',
                                                         repeat=100,
                                                         max_mutants_per_pool=1, max_workers=1):
        """X (max_mutants_per_pool) mutants are selected from each pool before passing to the next random one.
        The pools are traversed in a random order (different one for every repetition).
        1st degree order of pools is conserved.
        Once all pools are traversed, we restart from the first one again, picking X mutants by pool, etc.
        """
        result = load_cached_progress(cache_dir, tmp_file)

        if failing_test_ids is None:
            failing_tests = {t for p1 in mutant_pools for p in p1 for m in p for t in m.failing_tests}
            failing_test_ids = {t: i for i, t in enumerate(failing_tests)}
            m_ids_pools = [[[m.int_copy(failing_test_ids) for m in p] for p in p1] for p1 in mutant_pools]
            assert len([m for sp in m_ids_pools for p in sp for m in p if m.killed]) == len(
                [m for sp in mutant_pools for p in sp for m in p if m.killed])
        else:
            m_ids_pools = mutant_pools

        random_ordered_pools = []

        for _ in range(len(result), repeat):
            pools = []
            for p_pool in m_ids_pools:
                # we shuffle the order of the sub-pools inside of the large pools.
                random.shuffle(p_pool)
                pools.append(p_pool)
            random_ordered_pools.append(pools)
            # result.append(
            #     self.process_x_mutants_by_ranked_pool_by_ranked_pools(pools, failing_test_ids, ranked=False, repeat=1,
            #                                                           max_mutants_per_pool=max_mutants_per_pool))
            # save_zipped_pickle(result, join(cache_dir, tmp_file))
            # print('{0} simulation repetitions cached in {1}'.format(str(len(result)), tmp_file))
        result = process_parallel_run(self.process_x_mutants_by_ranked_pool_by_ranked_pools, random_ordered_pools,
                                      failing_test_ids, False, 1, max_mutants_per_pool, max_workers=max_workers,
                                      ignore_results=False)
        return result

    def process_x_mutants_by_random_pool(self, mutant_pools: List[List[Mutant]], failing_test_ids=None, repeat=100,
                                         max_mutants_per_pool=1, max_workers=1): #todo refactor
        """X (max_mutants_per_pool) mutants are selected from each pool before passing to the next random one.
        The pools are traversed in a random order (different one for every repetition).
        Once all pools are traversed, we restart from the first one again, picking X mutants by pool, etc.
        """
        result = []
        if failing_test_ids is None:
            failing_tests = {t for p in mutant_pools for m in p for t in m.failing_tests}
            failing_test_ids = {t: i for i, t in enumerate(failing_tests)}
            m_ids_pools = [[m.int_copy(failing_test_ids) for m in p1] for p1 in mutant_pools]
        else:
            m_ids_pools = mutant_pools

        random_ordered_pools = []
        for _ in range(0, repeat):
            random.shuffle(m_ids_pools)
            random_ordered_pools.append(m_ids_pools)

        result = process_parallel_run(self.process_x_mutants_by_ranked_pool, random_ordered_pools, failing_test_ids,
                                      False, 1,
                                      max_mutants_per_pool, max_workers=max_workers, ignore_results=False)
        # result.append(self.process_x_mutants_by_ranked_pool(m_ids_pools, failing_test_ids, ranked=False, repeat=1,
        #                                                     max_mutants_per_pool=max_mutants_per_pool))
        return result

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

    def process_x_mutants_by_random_pool_by_ranked_pools(self, mutant_pools: List[List[List[Mutant]]], repeat=100,
                                                         max_mutants_per_pool=1):
        """X (max_mutants_per_pool) mutants are selected from each pool before passing to the next random one.
        The pools are traversed in a random order (different one for every repetition).
        1st degree order of pools is conserved.
        Once all pools are traversed, we restart from the first one again, picking X mutants by pool, etc.
        """
        result = []
        for _ in range(0, repeat):
            pools = []
            random.shuffle(mutant_pools)  # shuffle lines
            for p_pool in mutant_pools:
                sh_p_pool = []
                # p_pool of mutants of the same line
                for m_pool in p_pool:
                    # m_pool of mutants with the same score
                    random.shuffle(m_pool)  # shuffle mutants with the same score
                    sh_p_pool.extend(m_pool)
                pools.append(sh_p_pool)
            result.append(self.process_x_mutants_by_ranked_pool(pools, ranked=True, repeat=1,
                                                                max_mutants_per_pool=max_mutants_per_pool))
        return result

    def process_x_mutants_by_ranked_pool_by_random_pool_by_ranked_pools(self,
                                                                        mutant_pools: List[List[List[List[Mutant]]]],
                                                                        repeat=100,
                                                                        max_mutants_per_pool=1):

        """
        1st level (simple or not): each pool is emptied before passing to the next one.
        2nd level (lines): ordered randomly, then x mutants is selected from each pool before passing to the next one,
        and we iterate.
        3rd level (scores): order conserved
        4th level (mutants): ordered randomly.

        """
        result = []
        for _ in range(0, repeat):
            ranked_pools = []
            for p_pool in mutant_pools:
                # p_pool of lines_scores_mutants form the same pattern: simple or not
                sub_ranked_pools = []
                # shuffle the lines order
                random.shuffle(p_pool)
                for sub_pool in p_pool:
                    # sub_pool of 1 line scores_mutants (of the same line).
                    line_pools = []
                    # we do not shuffle the scores order (sub_pool).
                    for score_pool in sub_pool:
                        # we shuffle the mutants of the same score
                        random.shuffle(score_pool)
                        # we add the mutants to the line
                        line_pools.extend(score_pool)
                    # we add the line to the pattern pool
                    sub_ranked_pools.append(line_pools)
                # we add the the pattern pool to the ranked_pools.
                ranked_pools.append(sub_ranked_pools)
            result.append(self.process_x_mutants_by_ranked_pool_by_ranked_pools(ranked_pools, ranked=True, repeat=1,
                                                                                max_mutants_per_pool=max_mutants_per_pool))
        return result

    def process_x_mutants_by_ranked_pool_by_ranked_pool_by_random_pools(self,
                                                                        mutant_pools: List[List[List[List[Mutant]]]],
                                                                        repeat=100,
                                                                        max_mutants_per_pool=1):
        """X (max_mutants_per_pool) mutants are selected from each pool before passing to the next random one.
        The pools are traversed in a random order (different one for every repetition).
        1st degree order of pools is conserved.
        Once all pools are traversed, we restart from the first one again, picking X mutants by pool, etc.
        """
        result = []
        for _ in range(0, repeat):
            ranked_pools = []
            # p_pool : simple or not
            for p_pool in mutant_pools:
                sub_ranked_pools = []
                # sub_pool: nat groups.
                for sub_pool in p_pool:
                    # shuffle which line to consider first
                    random.shuffle(sub_pool)
                    for line_pool in sub_pool:
                        sub_ranked_pools.append(line_pool)
                ranked_pools.append(sub_ranked_pools)
            result.append(self.process_x_mutants_by_ranked_pool_by_ranked_pools(ranked_pools, ranked=False, repeat=1,
                                                                                max_mutants_per_pool=max_mutants_per_pool))
        return result

    def process_x_mutants_by_ranked_pool(self, mutant_pools: List[List[Mutant]], failing_test_ids=None,
                                         ranked: bool = False, repeat=100,
                                         max_mutants_per_pool=1):
        """X (max_mutants_per_pool) mutants are selected from each pool before passing to the next one.
        The pools are traversed in the given order.
        Once all pools are traversed, we restart from the first one again, picking X mutants by pool, etc.
        ranked param controls the selection of mutants inside one pool."""
        if failing_test_ids is None:
            failing_tests = {t for p in mutant_pools for m in p for t in m.failing_tests}
            failing_test_ids = {t: i for i, t in enumerate(failing_tests)}
            m_ids_pools: List[List[Mutant]] = [[m.int_copy(failing_test_ids) for m in p1] for p1 in mutant_pools]
        else:
            m_ids_pools = mutant_pools

        if ranked:
            assert repeat <= 1, 'No repetition needed, if the pools and their mutants are ranked.'
            return [PractitionerXByPool(m_ids_pools, failing_test_ids, ranked,
                                        max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests)]
        else:
            return [PractitionerXByPool(m_ids_pools, failing_test_ids, ranked,
                                        max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests) for _ in range(0, repeat)]

    def process_x_mutants_by_ranked_pool_by_ranked_pools(self, mutant_pools: List[List[List[Mutant]]],
                                                         failing_test_ids=None,
                                                         ranked: bool = False, repeat=100, max_mutants_per_pool=1):
        """the higher level of pools or super_pools (each pool contains pools of mutants) are
        ranked and traversed in that order until finished:
        we finish all mutants of the pools of the super_pool before passing to the next super_pool.
        inside the super_pool, the pools traversing and mutants selection is similar to the one of
        process_x_mutants_by_ranked_pool().
        ranked param controls the selection of mutants inside one pool."""

        if failing_test_ids is None:
            failing_tests = {t for p1 in mutant_pools for p in p1 for m in p for t in m.failing_tests}
            failing_test_ids = {t: i for i, t in enumerate(failing_tests)}
            m_ids_pools = [[[m.int_copy(failing_test_ids) for m in p] for p in p1] for p1 in mutant_pools]
        else:
            m_ids_pools = mutant_pools

        if ranked:
            assert repeat <= 1, 'No repetition needed, if the pools and their mutants are ranked.'
            return [PractitionerXByPoolPoolByPool(m_ids_pools, failing_test_ids, ranked=ranked,
                                                  max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests)]
        else:
            return [PractitionerXByPoolPoolByPool(m_ids_pools, failing_test_ids, ranked=ranked,
                                                  max_mutants_per_pool=max_mutants_per_pool).simulate(
                self.failing_tests) for _ in range(0, repeat)]

    def process_ranked_pools(self, mutant_pools: List[List[Mutant]], failing_test_ids=None, repeat=100):
        result = []

        if failing_test_ids is None:
            failing_tests = {t for p in mutant_pools for m in p for t in m.failing_tests}
            failing_test_ids = {t: i for i, t in enumerate(failing_tests)}
            m_ids_pools = [[m.int_copy(failing_test_ids) for m in p1] for p1 in mutant_pools]
        else:
            m_ids_pools = mutant_pools

        for _ in range(0, repeat):
            mutants = []
            for m_pool in m_ids_pools:
                random.shuffle(m_pool)
                mutants.extend(m_pool)
            result.append(self.process(mutants, failing_test_ids, True, repeat=1))
        return result

    def process(self, mutants: List[Mutant], failing_test_ids=None, ranked: bool = False,
                repeat=100):  # default is random.
        if failing_test_ids is None:
            failing_tests = {t for m in mutants for t in m.failing_tests}
            failing_test_ids = {t: i for i, t in enumerate(failing_tests)}
            m_ids = [m.int_copy(failing_test_ids) for m in mutants]
        else:
            m_ids = mutants
        return [Practitioner(m_ids, failing_test_ids, ranked).simulate(self.failing_tests) for _ in range(0, repeat)]
