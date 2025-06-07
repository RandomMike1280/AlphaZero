import numpy as np
from .game import Game

class Reversi(Game):
    """
    Implementation of Reversi (Othello) game for AlphaZero.
    
    Board representation:
    - 0: Empty
    - 1: Player 1 (Black)
    - -1: Player -1 (White)
    """
    
    def __init__(self):
        """Initialize Reversi game parameters."""
        self.row_count = 8
        self.column_count = 8
        self.action_size = self.row_count * self.column_count
        
    def get_initial_state(self):
        """
        Returns the initial state of the Reversi board.
        
        Returns:
            8x8 numpy array with starting pieces in the center.
        """
        state = np.zeros((self.row_count, self.column_count))
        # Place the initial 4 pieces in the center
        mid_row, mid_col = self.row_count // 2, self.column_count // 2
        state[mid_row-1:mid_row+1, mid_col-1:mid_col+1] = np.array([[1, -1], [-1, 1]])
        return state
    
    def is_valid_move(self, state, action, player):
        """Check if a move is valid."""
        if action is None:
            return False
            
        row = action // self.column_count
        col = action % self.column_count
        
        # Check if the cell is empty
        if state[row, col] != 0:
            return False
            
        # Check all 8 directions
        directions = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),          (0,1),
                     (1,-1),  (1,0),  (1,1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            
            # Move in the direction until we find player's piece or hit the edge
            while 0 <= r < self.row_count and 0 <= c < self.column_count:
                if state[r, c] == 0:  # Empty cell
                    break
                elif state[r, c] == -player:  # Opponent's piece
                    to_flip.append((r, c))
                else:  # Player's piece
                    if to_flip:  # If we have pieces to flip
                        return True
                    break
                r += dr
                c += dc
        
        return False
    
    def get_valid_moves(self, state):
        """
        Returns a binary vector of valid moves for the current player.
        
        Args:
            state: Current 8x8 board state.
            
        Returns:
            Binary vector of length 64, where 1 means the move is valid.
        """
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        
        for action in range(self.action_size):
            row = action // self.column_count
            col = action % self.column_count
            if state[row, col] == 0 and self.is_valid_move(state, action, 1):
                valid_moves[action] = 1
                
        return valid_moves
    
    def get_next_state(self, state, action, player):
        """
        Applies the action to the current state and returns the new state.
        
        Args:
            state: Current 8x8 board state.
            action: Integer in [0, 63] representing position on the board.
            player: 1 or -1, representing the player.
            
        Returns:
            New state after the action is applied.
        """
        if action is None:  # Pass
            return state
            
        row = action // self.column_count
        col = action % self.column_count
        next_state = state.copy()
        next_state[row, col] = player
        
        # Flip opponent's pieces
        directions = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),          (0,1),
                     (1,-1),  (1,0),  (1,1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            
            # Move in the direction until we find player's piece or hit the edge
            while 0 <= r < self.row_count and 0 <= c < self.column_count:
                if next_state[r, c] == 0:  # Empty cell
                    break
                elif next_state[r, c] == -player:  # Opponent's piece
                    to_flip.append((r, c))
                else:  # Player's piece
                    for flip_r, flip_c in to_flip:
                        next_state[flip_r, flip_c] = player
                    break
                r += dr
                c += dc
                
        return next_state
    
    def check_win(self, state):
        """
        Check if the game is over and return the winner.
        
        Returns:
            1 if player 1 (Black) wins,
            -1 if player -1 (White) wins,
            0 for a draw,
            None if the game is not over
        """
        # Check if there are valid moves for either player
        if np.sum(self.get_valid_moves(state)) > 0 or np.sum(self.get_valid_moves(-state)) > 0:
            return None
            
        # Count pieces
        count = np.sum(state)
        if count > 0:
            return 1  # Black wins
        elif count < 0:
            return -1  # White wins
        else:
            return 0  # Draw
    
    def get_value_and_terminated(self, state, action):
        """
        Checks if the game is over and returns the value.
        
        Args:
            state: Current 8x8 board state.
            action: Last action taken.
            
        Returns:
            (value, terminated): 
                - value: 1 if player 1 wins, -1 if player -1 wins, 0 if draw.
                - terminated: True if the game is over, False otherwise.
        """
        winner = self.check_win(state)
        if winner is not None:
            return winner, True
        return 0, False
    
    def change_perspective(self, state, player):
        """
        Changes the perspective of the state to the given player.
        
        Args:
            state: Current 8x8 board state.
            player: Player to change perspective to (1 or -1).
            
        Returns:
            State from the perspective of the given player.
        """
        return state * player
    
    def get_encoded_state(self, state):
        """
        Encodes the state for neural network input.
        
        Args:
            state: Current 8x8 board state.
            
        Returns:
            Encoded state with shape (3, 8, 8):
                - Channel 0: Player 1's pieces (1 where player 1 has a piece, 0 elsewhere)
                - Channel 1: Empty spaces (1 where empty, 0 elsewhere)
                - Channel 2: Player -1's pieces (1 where player -1 has a piece, 0 elsewhere)
        """
        encoded_state = np.stack(
            (state == 1, state == 0, state == -1)
        ).astype(np.float32)
        
        # If we're dealing with a batch of states
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
            
        return encoded_state
    
    def get_symmetries(self, encoded_state, policy):
        """
        Generate symmetries (rotations, reflections) of the state and policy.
        
        Args:
            encoded_state: Encoded state with shape (3, 8, 8).
            policy: Action probabilities with shape (64,).
            
        Returns:
            List of (encoded_state, policy) tuples for each symmetry.
        """
        # Convert policy from vector to 8x8 grid
        policy_grid = policy.reshape(self.row_count, self.column_count)
        
        # For encoded_state, we need to work with the channels separately
        channel_0 = encoded_state[0]  # Player 1's pieces
        channel_1 = encoded_state[1]  # Empty spaces
        channel_2 = encoded_state[2]  # Player -1's pieces
        
        symmetries = []
        
        # Generate 4 rotated positions (0°, 90°, 180°, 270°)
        for i in range(4):
            # Rotate the current state and policy
            rot_channel_0 = np.rot90(channel_0, i)
            rot_channel_1 = np.rot90(channel_1, i)
            rot_channel_2 = np.rot90(channel_2, i)
            rot_policy = np.rot90(policy_grid, i)
            
            # Create new encoded state from rotated channels
            rot_encoded_state = np.stack([rot_channel_0, rot_channel_1, rot_channel_2])
            
            # Add the rotated position
            symmetries.append((rot_encoded_state, rot_policy.flatten()))
            
            # Also add flipped version of this rotation
            flip_channel_0 = np.fliplr(rot_channel_0)
            flip_channel_1 = np.fliplr(rot_channel_1)
            flip_channel_2 = np.fliplr(rot_channel_2)
            flip_policy = np.fliplr(rot_policy)
            
            # Create new encoded state from flipped channels
            flip_encoded_state = np.stack([flip_channel_0, flip_channel_1, flip_channel_2])
            
            # Add the flipped position
            symmetries.append((flip_encoded_state, flip_policy.flatten()))
        
        return symmetries
