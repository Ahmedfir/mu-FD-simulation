from ms.tool_mutants import ToolMutants


class MutantsProvider:
    def __init__(self, pid_bid):
        self.pid_bid = pid_bid

    def get_mutants(self, tool_name) -> ToolMutants:
        pass