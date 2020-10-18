import numpy as np
import argparse
from encoder import Maze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid',type=str,help='Path to maze file')
    parser.add_argument('--value_policy',type=str,help='Path to value and policy file')
    args = parser.parse_args()
    maze = Maze()
    maze.parse(args.grid,partial=True)
    maze.solve_from_file(args.value_policy)
    path = maze.get_path()
    print(" ".join(path))
