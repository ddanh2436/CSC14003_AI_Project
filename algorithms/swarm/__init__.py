# Import các class từ các file thành phần
from .aco import AntColonyOptimization
from .pso import ParticleSwarmOptimization
from .abc import ArtificialBeeColony
from .fa import FireflyAlgorithm
from .cs import CuckooSearch

# --- TẠO ALIAS (TÊN VIẾT TẮT) ---
# Mục đích: Giúp main.py có thể gọi ngắn gọn: "from algorithms.swarm import PSO"
ACO = AntColonyOptimization
PSO = ParticleSwarmOptimization
ABC = ArtificialBeeColony
FA = FireflyAlgorithm
CS = CuckooSearch

# (Tùy chọn) Định nghĩa danh sách các module được export khi dùng "from algorithms.swarm import *"
__all__ = [
    'AntColonyOptimization', 'ACO',
    'ParticleSwarmOptimization', 'PSO',
    'ArtificialBeeColony', 'ABC',
    'FireflyAlgorithm', 'FA',
    'CuckooSearch', 'CS'
]