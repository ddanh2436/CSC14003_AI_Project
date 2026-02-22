from .a_star import AStarSearch
from .bfs import BreadthFirstSearch
from .dfs import DepthFirstSearch
from .gbf import GreedyBestFirstSearch
from .hill_climbing import HillClimbing
from .hill_climbing_tsp import HillClimbingTSP
from .ucs import UniformCostSearch

AS = AStarSearch
BFS = BreadthFirstSearch
DFS = DepthFirstSearch
UCS = UniformCostSearch
GBF = GreedyBestFirstSearch
HC = HillClimbing
HCT = HillClimbingTSP
SA = SimulatedAnnealing

__all__ = [
    'AStarSearch', 'AS',
    'BreadthFirstSearch', 'BFS',
    'DepthFirstSearch', 'DFS',
    'UniformCostSearch', 'UCS',
    'GreedyBestFirstSearch', 'GBF',
    'HillClimbing', 'HC',
    'HillClimbingTSP', 'HCT',
    'SimulatedAnnealing' 'SA',
]