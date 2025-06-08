import numpy as np
from typing import List, Tuple, Optional

# Assuming a base Game class exists, like:
# class Game:
#     def get_action_size(self): pass
#     ...
# We'll make our own simple one for this standalone example.
class Game:
    def __init__(self):
        self.action_size = 0

class Reversi(Game):
    """
    An optimized implementation of the Reversi (Othello) game environment.

    Board representation:
    - 1: Player 1 (Black)
    - -1: Player -1 (White)
    - 0: Empty
    """
    # Class constants for directions and board size for clarity and efficiency
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    def __init__(self, board_size: int = 8):
        """Initialize Reversi game parameters."""
        super().__init__()
        assert board_size % 2 == 0, "Board size must be an even number."
        self.row_count = board_size
        self.column_count = board_size
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self) -> np.ndarray:
        """Returns the initial state of the Reversi board."""
        state = np.zeros((self.row_count, self.column_count), dtype=np.int8)
        mid = self.row_count // 2
        state[mid - 1, mid - 1] = 1
        state[mid - 1, mid] = -1
        state[mid, mid - 1] = -1
        state[mid, mid] = 1
        return state

    def _get_flips_for_move(self, state: np.ndarray, action: int, player: int) -> List[Tuple[int, int]]:
        """
        Calculates all opponent pieces that would be flipped by a given move.
        This is a core helper function to avoid code duplication.
        
        Returns an empty list if the move is invalid.
        """
        row, col = action // self.column_count, action % self.column_count

        # Move is invalid if the square is not empty
        if state[row, col] != 0:
            return []

        pieces_to_flip = []
        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            line = []
            while 0 <= r < self.row_count and 0 <= c < self.column_count:
                if state[r, c] == -player: # Opponent's piece
                    line.append((r, c))
                elif state[r, c] == player: # Player's own piece
                    pieces_to_flip.extend(line)
                    break
                else: # Empty square
                    break
                r += dr
                c += dc
        return pieces_to_flip

    def get_valid_moves(self, state: np.ndarray, player: int) -> np.ndarray:
        """
        Returns a binary vector of valid moves for the given player.
        """
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        # Find all empty squares
        empty_squares = np.where(state == 0)
        for r, c in zip(*empty_squares):
            action = r * self.column_count + c
            # A move is valid if it flips at least one piece
            if self._get_flips_for_move(state, action, player):
                valid_moves[action] = 1
        return valid_moves

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """
        Applies the action to the current state and returns the new state.
        Assumes the action is valid.
        """
        next_state = state.copy()
        row, col = action // self.column_count, action % self.column_count
        
        pieces_to_flip = self._get_flips_for_move(state, action, player)
        
        # Place the new piece and flip the opponent's pieces
        next_state[row, col] = player
        for r, c in pieces_to_flip:
            next_state[r, c] = player
            
        return next_state

    def get_value_and_terminated(self, state: np.ndarray, player: int) -> Tuple[float, bool]:
        """
        Checks if the game has ended and returns the value.
        The value is from the perspective of the *next* player to move.
        """
        # Check if the current player has any valid moves
        if np.any(self.get_valid_moves(state, player)):
            return 0.0, False

        # If the current player cannot move, check the opponent
        if np.any(self.get_valid_moves(state, -player)):
            return 0.0, False

        # If neither player can move, the game is over
        piece_count = np.sum(state)
        if piece_count == 0:
            return 0.0, True  # Draw
        
        # Winner is the one with more pieces. The value is relative to the current player.
        # If player (e.g. 1) is the winner (piece_count > 0), value is 1.
        # If player (e.g. -1) is the winner (piece_count < 0), value is 1.
        winner = np.sign(piece_count)
        return 1.0 if winner == player else -1.0, True

    def get_opponent(self, player: int) -> int:
        return -player

    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        """Changes the perspective of the state to the given player."""
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """
        Encodes the state for neural network input.
        - Channel 0: Player's pieces
        - Channel 1: Opponent's pieces
        - Channel 2: Empty spaces (optional, but can be useful)
        """
        return np.stack([
            (state == 1).astype(np.float32),
            (state == -1).astype(np.float32),
        ])
        # A 3-channel version could be:
        # return np.stack([
        #     (state == 1).astype(np.float32), 
        #     (state == -1).astype(np.float32),
        #     (state == 0).astype(np.float32)
        # ])

    def get_symmetries(self, state_encoded: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates symmetries (rotations and reflections) of the state and policy.
        """
        policy_grid = policy.reshape(self.row_count, self.column_count)
        symmetries = []

        for i in range(4):  # Rotations: 0, 90, 180, 270 degrees
            # Rotate state tensor and policy grid
            # axes=(1, 2) rotates the spatial dimensions of the (C, H, W) tensor
            rot_state = np.rot90(state_encoded, k=i, axes=(1, 2))
            rot_policy = np.rot90(policy_grid, k=i)
            symmetries.append((rot_state, rot_policy.flatten()))
            
            # Flipped versions of the rotated states
            flip_state = np.fliplr(rot_state)
            flip_policy = np.fliplr(rot_policy)
            symmetries.append((flip_state, flip_policy.flatten()))
            
        return symmetries