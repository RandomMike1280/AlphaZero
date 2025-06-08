# reversi.py

import numpy as np
from typing import List, Tuple
from .game import Game # Relative import to link with game.py

class Reversi(Game):
    """
    An optimized implementation of the Reversi (Othello) game environment,
    conforming to the Game abstract base class.

    Board representation:
    - 1: Player 1 (Black)
    - -1: Player -1 (White)
    - 0: Empty
    """
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                  (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, board_size: int = 8):
        super().__init__()
        assert board_size % 2 == 0, "Board size must be an even number."
        self.row_count = board_size
        self.column_count = board_size
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self) -> np.ndarray:
        state = np.zeros((self.row_count, self.column_count), dtype=np.int8)
        mid = self.row_count // 2
        state[mid - 1, mid - 1] = 1
        state[mid - 1, mid] = -1
        state[mid, mid - 1] = -1
        state[mid, mid] = 1
        return state

    def _get_flips_for_move(self, state: np.ndarray, action: int, player: int) -> List[Tuple[int, int]]:
        """Helper to calculate all pieces that would be flipped by a move."""
        row, col = action // self.column_count, action % self.column_count
        if state[row, col] != 0:
            return []

        pieces_to_flip = []
        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            line = []
            while 0 <= r < self.row_count and 0 <= c < self.column_count:
                if state[r, c] == -player:
                    line.append((r, c))
                elif state[r, c] == player:
                    pieces_to_flip.extend(line)
                    break
                else: # Empty square
                    break
                r += dr
                c += dc
        return pieces_to_flip

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a binary vector of valid moves for the current player.
        Note: This implementation assumes the state is from the perspective
        of the current player, who is always represented by 1.
        """
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        empty_squares_r, empty_squares_c = np.where(state == 0)
        
        for r, c in zip(empty_squares_r, empty_squares_c):
            action = r * self.column_count + c
            if self._get_flips_for_move(state, action, player=1):
                valid_moves[action] = 1
        return valid_moves
    
    def _has_valid_moves(self, state: np.ndarray, player: int) -> bool:
        """A more efficient check if any valid moves exist for a player."""
        empty_squares_r, empty_squares_c = np.where(state == 0)
        for r, c in zip(empty_squares_r, empty_squares_c):
            action = r * self.column_count + c
            if self._get_flips_for_move(state, action, player):
                return True
        return False

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        next_state = state.copy()
        row, col = action // self.column_count, action % self.column_count
        
        pieces_to_flip = self._get_flips_for_move(state, action, player)
        
        next_state[row, col] = player
        for r, c in pieces_to_flip:
            next_state[r, c] = player
            
        return next_state

    def check_win(self, state: np.ndarray, action: int) -> bool:
        """
        Checks if the game has ended with a winner (not a draw).
        The 'action' parameter is not used in Reversi's end-game check.
        This function is less informative than get_value_and_terminated.
        """
        # Game ends if neither player has a valid move.
        if self._has_valid_moves(state, 1) or self._has_valid_moves(state, -1):
            return False  # Game is not over
        
        # Game is over, check for a non-draw winner
        return np.sum(state) != 0

    def get_value_and_terminated(self, state: np.ndarray, action: int) -> Tuple[float, bool]:
        """
        Checks if the game has ended and returns the value.
        The value is from the perspective of the player who *just moved* to
        create the given state.
        The 'action' parameter is not used.
        """
        # Game ends if the NEXT player to move has no moves, and the current player also has no moves.
        # Let's assume the state is from a canonical perspective (1=Black, -1=White).
        # We check if player 1 has moves, then if player -1 has moves.
        if self._has_valid_moves(state, 1) or self._has_valid_moves(state, -1):
            return 0.0, False

        # If neither player can move, the game is over
        piece_count = np.sum(state)
        if piece_count == 0:
            return 0.0, True  # Draw

        # There is a winner. The value is relative to the player who just moved.
        # This function doesn't know who just moved, which is a limitation of the ABC.
        # A common convention is to return the game's outcome (1 for Black win, -1 for White win).
        return np.sign(piece_count), True

    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """
        Encodes the state for neural network input from the perspective of player 1.
        - Channel 0: Player 1's pieces (Black)
        - Channel 1: Player -1's pieces (White)
        """
        return np.stack([
            (state == 1).astype(np.float32),
            (state == -1).astype(np.float32),
            (state == 0).astype(np.float32)
        ])

    def get_symmetries(self, state_encoded: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates symmetries (8 rotations and reflections) of the state and policy.
        This is a utility function for training, not part of the core ABC.
        """
        policy_grid = policy.reshape(self.row_count, self.column_count)
        symmetries = []

        for i in range(4):  # Rotations: 0, 90, 180, 270 degrees
            # Rotate state tensor and policy grid
            rot_state = np.rot90(state_encoded, k=i, axes=(1, 2)).copy()
            rot_policy = np.rot90(policy_grid, k=i)
            symmetries.append((rot_state, rot_policy.flatten()))
            
            # Flipped versions of the rotated states
            # Use np.flip over an axis for tensors
            flip_state = np.flip(rot_state, axis=2).copy()
            flip_policy = np.fliplr(rot_policy)
            symmetries.append((flip_state, flip_policy.flatten()))
            
        return symmetries