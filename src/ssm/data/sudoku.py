from copy import deepcopy
from flax import struct
import jax.numpy as jnp


class Sudoku:
    
    def __init__(self, puzzle):
        self.width = puzzle.width
        self.size = self.width ** 2
        self.zeros = [0] * self.size
        self.map = self._make_map(self.size)
        self.puzzle = puzzle
        self.solution = puzzle.solve()
        self.solution_tensor = self.to_tensor()
        self.mask = self.make_mask()
        
    def _make_map(self, size):
        kvs = {}
        for i in range(size):
            l = deepcopy(self.zeros)
            l[i] = 1
            kvs[i+1] = l
        return kvs

    def make_mask(self):
        board = []
        for row in self.puzzle.board:
            for value in row:
                if value is not None:
                    board.append(0)
                else:
                    board.append(1)
        return jnp.array(board)
                
    def to_tensor(self):
        board = []
        for row in self.solution.board:
            for value in row:
                board.append(self.map[value])
        return jnp.array(board)

    @classmethod
    def from_jax(cls, flat_board):
        pass


@struct.dataclass
class SudokuRandomWalker:

