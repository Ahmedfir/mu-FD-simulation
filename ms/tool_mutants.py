from typing import List

from mutant import Mutant


class ToolMutants:

    def __init__(self, bug: str, tool: str, all_mutants: List[Mutant], subsuming_mutants: List[Mutant]):
        self.bug: str = bug
        self.tool: str = tool
        self.killable_mutants: List[Mutant] = [m for m in all_mutants if m.killed]
        self.all_mutants_l: int = len(all_mutants)
        self.subsuming_mutants: List[Mutant] = subsuming_mutants
