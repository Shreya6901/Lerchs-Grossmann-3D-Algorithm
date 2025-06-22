import numpy as np
import pandas as pd
from collections import defaultdict, deque
import heapq
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Block:
    """Represents a mining block with economic and geometric properties"""
    
    def __init__(self, x: int, y: int, z: int, value: float, 
                 ore_grade: float = 0.0, rock_type: str = "waste"):
        self.x = x
        self.y = y  
        self.z = z
        self.value = value  # Net economic value (revenue - cost)
        self.ore_grade = ore_grade
        self.rock_type = rock_type
        self.extracted = False
        self.id = f"{x}_{y}_{z}"
    
    def __repr__(self):
        return f"Block({self.x},{self.y},{self.z},${self.value:.2f})"

class LerchsGrossmann3D:
    """3D Lerchs-Grossmann algorithm for ultimate pit limit optimization"""

    def load_from_csv(self, filename: str):
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            self.add_block(
                int(row["X"]), int(row["Y"]), int(row["Z"]),
                float(row["Value"]), float(row["Grade"]),
                row["Rock_Type"]
            )

    
    def __init__(self, nx: int, ny: int, nz: int, slope_angles: Dict[str, float] = None): # type: ignore
        self.nx = nx  # Grid dimensions
        self.ny = ny
        self.nz = nz
        self.blocks = {}  # Dictionary to store blocks by (x,y,z)
        self.precedence_graph = defaultdict(list)  # Precedence relationships
        self.reverse_graph = defaultdict(list)     # Reverse precedence
        
        # Default slope angles in degrees for different directions
        self.slope_angles = slope_angles or {
            'overall': 45.0,  # Overall slope angle
            'bench': 70.0,    # Individual bench slope
        }
        
        # Convert to slope ratios (horizontal:vertical)
        self.slope_ratios = {
            k: 1.0 / np.tan(np.radians(v)) for k, v in self.slope_angles.items()
        }
    
    def add_block(self, x: int, y: int, z: int, value: float, 
                  ore_grade: float = 0.0, rock_type: str = "waste"):
        """Add a block to the model"""
        if 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz:
            block = Block(x, y, z, value, ore_grade, rock_type)
            self.blocks[(x, y, z)] = block
            return block
        else:
            raise ValueError(f"Block coordinates ({x},{y},{z}) out of bounds")
    
    def generate_random_model(self, ore_probability: float = 0.3, 
                            ore_value_range: Tuple[float, float] = (50, 150),
                            waste_cost: float = -10):
        """Generate a random block model for testing"""
        np.random.seed(42)  # For reproducible results
        
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    # Higher probability of ore at deeper levels
                    depth_factor = (self.nz - z) / self.nz
                    adj_ore_prob = ore_probability * (1 + depth_factor)
                    
                    if np.random.random() < adj_ore_prob:
                        # Ore block
                        value = np.random.uniform(*ore_value_range)
                        grade = np.random.uniform(0.5, 3.0)
                        self.add_block(x, y, z, value, grade, "ore")
                    else:
                        # Waste block
                        self.add_block(x, y, z, waste_cost, 0.0, "waste")
    
    def build_precedence_constraints(self):
        """Build precedence constraints based on slope stability"""
        # Clear existing constraints
        self.precedence_graph.clear()
        self.reverse_graph.clear()
        
        slope_ratio = self.slope_ratios['overall']
        
        for (x, y, z), block in self.blocks.items():
            # For each block, find all blocks that must be removed first
            # This includes blocks above and within the slope cone
            
            for dx in range(-int(slope_ratio * (self.nz - z)), 
                           int(slope_ratio * (self.nz - z)) + 1):
                for dy in range(-int(slope_ratio * (self.nz - z)), 
                               int(slope_ratio * (self.nz - z)) + 1):
                    for dz in range(1, self.nz - z):  # Only blocks above
                        pred_x, pred_y, pred_z = x + dx, y + dy, z + dz
                        
                        # Check if predecessor is within bounds and slope constraint
                        if (0 <= pred_x < self.nx and 0 <= pred_y < self.ny and 
                            0 <= pred_z < self.nz):
                            
                            # Check slope constraint
                            horizontal_dist = np.sqrt(dx*dx + dy*dy)
                            vertical_dist = dz
                            
                            if horizontal_dist <= slope_ratio * vertical_dist:
                                if (pred_x, pred_y, pred_z) in self.blocks:
                                    # pred_block must be removed before current block
                                    pred_key = (pred_x, pred_y, pred_z)
                                    curr_key = (x, y, z)
                                    
                                    self.precedence_graph[pred_key].append(curr_key)
                                    self.reverse_graph[curr_key].append(pred_key)
    
    def solve_maximum_closure(self) -> Tuple[Set[Tuple[int, int, int]], float]:
        """
        Solve the maximum closure problem using a modified max-flow approach
        Returns the set of blocks to extract and total value
        """
        # Create source and sink nodes
        SOURCE = "SOURCE"
        SINK = "SINK"
        
        # Build flow network
        # Positive value blocks connect to source
        # Negative value blocks connect to sink
        # Precedence constraints become infinite capacity edges
        
        graph = defaultdict(lambda: defaultdict(float))
        
        # Add edges from source to positive blocks
        # Add edges from negative blocks to sink
        for (x, y, z), block in self.blocks.items():
            block_key = (x, y, z)
            
            if block.value > 0:
                graph[SOURCE][block_key] = block.value
            else:
                graph[block_key][SINK] = -block.value
        
        # Add precedence constraints as infinite capacity edges
        INF = float('inf')
        for pred_key, succ_list in self.precedence_graph.items():
            for succ_key in succ_list:
                if pred_key in self.blocks and succ_key in self.blocks:
                    graph[pred_key][succ_key] = INF
        
        # Run max flow algorithm (Ford-Fulkerson with BFS)
        max_flow_value = self._max_flow(graph, SOURCE, SINK)
        
        # Find minimum cut to determine which blocks to extract
        extracted_blocks = self._find_min_cut(graph, SOURCE, SINK)
        
        # Calculate total value
        total_value = sum(self.blocks[block_key].value 
                         for block_key in extracted_blocks 
                         if block_key in self.blocks)
        
        return extracted_blocks, total_value
    
    def _max_flow(self, graph: Dict, source: str, sink: str) -> float:
        """Implement Ford-Fulkerson algorithm with BFS (Edmonds-Karp)"""
        def bfs_find_path():
            visited = set([source])
            queue = deque([(source, [source])])
            
            while queue:
                node, path = queue.popleft()
                
                for neighbor in graph[node]:
                    if neighbor not in visited and graph[node][neighbor] > 0:
                        new_path = path + [neighbor]
                        if neighbor == sink:
                            return new_path
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
            return None
        
        max_flow = 0
        
        while True:
            path = bfs_find_path()
            if not path:
                break
            
            # Find minimum capacity along the path
            flow = float('inf')
            for i in range(len(path) - 1):
                flow = min(flow, graph[path[i]][path[i + 1]])
            
            # Update residual capacities
            for i in range(len(path) - 1):
                graph[path[i]][path[i + 1]] -= flow
                graph[path[i + 1]][path[i]] += flow
            
            max_flow += flow
        
        return max_flow
    
    def _find_min_cut(self, graph: Dict, source: str, sink: str) -> Set:
        """Find minimum cut using BFS from source"""
        visited = set()
        queue = deque([source])
        visited.add(source)
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited and graph[node][neighbor] > 0:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Return all blocks reachable from source (except source itself)
        return {node for node in visited if isinstance(node, tuple)}
    
    def optimize_pit(self) -> Dict:
        """Main optimization function"""
        print("Building precedence constraints...")
        self.build_precedence_constraints()
        
        print("Solving maximum closure problem...")
        extracted_blocks, total_value = self.solve_maximum_closure()
        
        # Mark extracted blocks
        for block_key in extracted_blocks:
            if block_key in self.blocks:
                self.blocks[block_key].extracted = True
        
        # Calculate statistics
        total_blocks = len(self.blocks)
        extracted_count = len(extracted_blocks)
        ore_blocks = sum(1 for b in self.blocks.values() if b.rock_type == "ore" and b.extracted)
        waste_blocks = extracted_count - ore_blocks
        
        return {
            'extracted_blocks': extracted_blocks,
            'total_value': total_value,
            'total_blocks': total_blocks,
            'extracted_count': extracted_count,
            'ore_blocks': ore_blocks,
            'waste_blocks': waste_blocks,
            'extraction_ratio': extracted_count / total_blocks if total_blocks > 0 else 0
        }
    
    def visualize_pit(self, show_all: bool = False):
        """Visualize the optimized pit in 3D"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        
        extracted_ore = []
        extracted_waste = []
        remaining_ore = []
        remaining_waste = []
        
        for (x, y, z), block in self.blocks.items():
            if block.extracted:
                if block.rock_type == "ore":
                    extracted_ore.append((x, y, z))
                else:
                    extracted_waste.append((x, y, z))
            elif show_all:
                if block.rock_type == "ore":
                    remaining_ore.append((x, y, z))
                else:
                    remaining_waste.append((x, y, z))
        
        # Plot extracted blocks
        if extracted_ore:
            x_ore, y_ore, z_ore = zip(*extracted_ore)
            ax.scatter(x_ore, y_ore, z_ore, c='blue', alpha=0.8, label='Extracted Ore')
        
        if extracted_waste:
            x_waste, y_waste, z_waste = zip(*extracted_waste)
            ax.scatter(x_waste, y_waste, z_waste, c='grey', alpha=0.6, label='Extracted Waste')
        
        # Plot remaining blocks if requested
        if show_all:
            if remaining_ore:
                x_rem_ore, y_rem_ore, z_rem_ore = zip(*remaining_ore)
                ax.scatter(x_rem_ore, y_rem_ore, z_rem_ore, c='green', alpha=0.3, label='Remaining Ore')
            
            if remaining_waste:
                x_rem_waste, y_rem_waste, z_rem_waste = zip(*remaining_waste)
                ax.scatter(x_rem_waste, y_rem_waste, z_rem_waste, c='red', alpha=0.2, label='Remaining Waste')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Depth)')
        ax.set_title('3D Mine Pit Optimization - Lerchs-Grossmann Algorithm')
        ax.legend()
        
        # Invert Z-axis to show depth properly
        ax.invert_zaxis()
        
        plt.tight_layout()
        plt.show()

    # def visualize_pit(self, show_all: bool = False):
    

    #     fig = plt.figure(figsize=(12, 10))
    #     ax = fig.add_subplot(111, projection='3d')

    #     shape = (self.nx, self.ny, self.nz)
    #     filled = np.zeros(shape, dtype=bool)
    #     colors = np.empty(shape, dtype=object)

    #     for (x, y, z), block in self.blocks.items():
    #         if block.extracted or show_all:
    #             filled[x, y, z] = True

    #             if block.extracted:
    #                 if block.rock_type == "ore":
    #                     colors[x, y, z] = 'gold'
    #                 else:
    #                     colors[x, y, z] = 'saddlebrown'
    #             else:
    #                 if block.rock_type == "ore":
    #                     colors[x, y, z] = 'lightyellow'
    #                 else:
    #                     colors[x, y, z] = 'lightgray'

    #     ax.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.1)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z (Depth)')
    #     ax.set_title('3D Pit Visualization (Voxels)')
    #     ax.invert_zaxis()
    #     plt.tight_layout()
    #     plt.show()

    
    def export_results(self, filename: str):
        """Export optimization results to CSV"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'Z', 'Value', 'Grade', 'Rock_Type', 'Extracted'])
            
            for (x, y, z), block in self.blocks.items():
                writer.writerow([x, y, z, block.value, block.ore_grade, 
                               block.rock_type, block.extracted])

# Example usage and testing
if __name__ == "__main__":
    # Create a mine model
    print("Creating 3D mine model...")
    mine = LerchsGrossmann3D(nx=20, ny=20, nz=15)

    # Read csv file
    mine.load_from_csv("mine_pit_dataset.csv")

    
    print(f"Generated {len(mine.blocks)} blocks")
    
    # Optimize the pit
    print("\nOptimizing pit limits...")
    results = mine.optimize_pit()
    
    # Print results
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Total economic value: ${results['total_value']:,.2f}")
    print(f"Blocks to extract: {results['extracted_count']:,} / {results['total_blocks']:,}")
    print(f"Ore blocks: {results['ore_blocks']:,}")
    print(f"Waste blocks: {results['waste_blocks']:,}")
    print(f"Extraction ratio: {results['extraction_ratio']:.1%}")
    
    # Visualize results
    print("\nGenerating visualization...")
    mine.visualize_pit(show_all=True)
    
    # Export results
    mine.export_results("pit_optimization_results.csv")
    print("Results exported to pit_optimization_results.csv")

# for this code, generate the dataset for mine pit 
