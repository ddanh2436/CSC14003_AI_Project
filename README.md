<div align="center">
  <svg width="850" height="60" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="rainbow_7_colors" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#FF0000">
          <animate attributeName="stop-color" values="#FF0000;#FF7F00;#FFFF00;#00FF00;#0000FF;#4B0082;#9400D3;#FF0000" dur="7s" repeatCount="indefinite" />
        </stop>
        <stop offset="14%" stop-color="#FF7F00">
          <animate attributeName="stop-color" values="#FF7F00;#FFFF00;#00FF00;#0000FF;#4B0082;#9400D3;#FF0000;#FF7F00" dur="7s" repeatCount="indefinite" />
        </stop>
        <stop offset="28%" stop-color="#FFFF00">
          <animate attributeName="stop-color" values="#FFFF00;#00FF00;#0000FF;#4B0082;#9400D3;#FF0000;#FF7F00;#FFFF00" dur="7s" repeatCount="indefinite" />
        </stop>
        <stop offset="42%" stop-color="#00FF00">
          <animate attributeName="stop-color" values="#00FF00;#0000FF;#4B0082;#9400D3;#FF0000;#FF7F00;#FFFF00;#00FF00" dur="7s" repeatCount="indefinite" />
        </stop>
        <stop offset="56%" stop-color="#0000FF">
          <animate attributeName="stop-color" values="#0000FF;#4B0082;#9400D3;#FF0000;#FF7F00;#FFFF00;#00FF00;#0000FF" dur="7s" repeatCount="indefinite" />
        </stop>
        <stop offset="70%" stop-color="#4B0082">
          <animate attributeName="stop-color" values="#4B0082;#9400D3;#FF0000;#FF7F00;#FFFF00;#00FF00;#0000FF;#4B0082" dur="7s" repeatCount="indefinite" />
        </stop>
        <stop offset="85%" stop-color="#9400D3">
          <animate attributeName="stop-color" values="#9400D3;#FF0000;#FF7F00;#FFFF00;#00FF00;#0000FF;#4B0082;#9400D3" dur="7s" repeatCount="indefinite" />
        </stop>
      </linearGradient>
    </defs>
    <text x="50%" y="45" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="bold" fill="url(#rainbow_7_colors)" text-anchor="middle">
      TEAM PROJECT 01 - SEARCH & NATURE-INSPIRED ALGORITHMS
    </text>
  </svg>
</div>

<span style="color:gold">Group: 7</span>

Repository link: [Group 7](https://github.com/ddanh2436/CSC14003_AI_Project.git)

I. Project Introduction

    🛫 This project is a comprehensive framework developed for the AI Course (CSC14003). It focuses on the implementation, evaluation, and benchmarking of various search and optimization algorithms. The system is designed to handle both continuous and discrete problems, providing a robust platform to compare performance metrics such as fitness accuracy, execution time, and memory consumption.

    💻 The framework supports advanced features like 3D landscape visualization, statistical hypothesis testing (T-Tests), and a unique historical simulation using the Imperialist Competitive Algorithm (ICA).

II. Group members

    Member 1: Đào Duy Anh (Student ID: 24127012) - Group Leader
    Member 2: Trần Huỳnh Mạnh Đạt (Student ID: 24127024)
    Member 3: Trần Hoàng Quốc Khánh (Student ID: 24127057)
    Member 4: Hồ Phúc Kiên (Student ID: 24127067)

III. Project details

    📂 Directory Layout
        - algorithms/: Core implementations.
            . classical/: BFS, DFS, A*, Hill Climbing, and Simulated Annealing.
            . evolutionary/: Genetic Algorithm (GA), Differential Evolution (DE), and Evolution Strategy (ES).
            . swarm/: PSO, Ant Colony (ACO), Artificial Bee Colony (ABC), Firefly, and Cuckoo Search.
            . human_based/: Teaching-Learning-Based Optimization (TLBO) and World History ICA.

        - problems/: Benchmarking environments.
            . continuous.py: Sphere, Rastrigin, Rosenbrock, and Ackley functions.
            . discrete.py: Traveling Salesman Problem (TSP), Knapsack, and Shortest Path.

        - utils/: Toolsets for analysis.
            . advanced_analysis.py: Statistical testing (T-Test, Wilcoxon) and Diversity analysis.
            . metrics.py: Performance measurement (Time & RAM tracking).
            . visualization.py: 3D surfaces and convergence plots.

    🚀 Key Features
        - Performance Metrics: Integrated memory tracking using tracemalloc to monitor peak RAM usage.

        - Statistical Validation: Built-in T-Test and Wilcoxon rank-sum tests to ensure results are statistically significant.

        - Visual Demonstration:
            . 3D Landscapes: Visualizing the search space of objective functions.
            . Graph Animations: Step-by-step animation for BFS and A* Search.
            . World History Simulation: A specialized ICA variant that simulates historical events (e.g., Rise of Empires, Fall of Rome) through optimization iterations.

    🛠 Installation & Usage

        1. Requirements: Install dependencies via pip:
            pip install -r requirements.txt
        *(Main libraries: numpy, matplotlib, seaborn, pandas)*
        
        2. Main Execution: Run the benchmark suite:
            python main.py
        
        3. Historical Simulation: Experience the ICA timeline visualization:
            python run_history.py