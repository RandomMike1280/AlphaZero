import numpy as np
from .game import Game


class TicTacToe(Game):
    """
    Implementation of Tic Tac Toe game for AlphaZero.
    
    Board representation:
    - 0: Empty
    - 1: Player 1 (X)
    - -1: Player -1 (O)
    """
    
    def __init__(self):
        """Initialize Tic Tac Toe game parameters."""
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        
    def get_initial_state(self):
        """
        Returns an empty 3x3 board.
        
        Returns:
            3x3 numpy array of zeros.
        """
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        """
        Places the player's mark at the specified action position.
        
        Args:
            state: Current 3x3 board state.
            action: Integer in [0, 8] representing position on the board.
            player: 1 or -1, representing the player.
            
        Returns:
            New state after the action is applied.
        """
        row = action // self.column_count
        column = action % self.column_count
        
        # Create a copy to avoid modifying the original state
        next_state = state.copy()
        next_state[row, column] = player
        return next_state
    
    def get_valid_moves(self, state):
        """
        Returns a binary vector of valid moves (empty cells).
        
        Args:
            state: Current 3x3 board state.
            
        Returns:
            Binary vector of length 9, where 1 means the move is valid.
        """
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        """
        Checks if the last action led to a win.
        
        Args:
            state: Current 3x3 board state.
            action: Last action taken (position on the board).
            
        Returns:
            True if the last action led to a win, False otherwise.
        """
        if action is None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        # Check row
        if np.sum(state[row, :]) == player * self.column_count:
            return True
        
        # Check column
        if np.sum(state[:, column]) == player * self.row_count:
            return True
        
        # Check main diagonal
        if row == column and np.sum(np.diag(state)) == player * self.row_count:
            return True
        
        # Check other diagonal
        if row + column == self.row_count - 1 and np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count:
            return True
        
        return False
    
    def get_value_and_terminated(self, state, action):
        """
        Checks if the game is over and returns the value.
        
        Args:
            state: Current 3x3 board state.
            action: Last action taken.
            
        Returns:
            (value, terminated): 
                - value: 1 if win, 0 if draw or game is not over.
                - terminated: True if the game is over, False otherwise.
        """
        if self.check_win(state, action):
            return 1, True
        
        # Check for draw
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        
        return 0, False
    
    def change_perspective(self, state, player):
        """
        Changes the perspective of the state to the given player.
        
        Args:
            state: Current 3x3 board state.
            player: Player to change perspective to (1 or -1).
            
        Returns:
            State from the perspective of the given player.
        """
        return state * player
    
    def get_encoded_state(self, state):
        """
        Encodes the state for neural network input.
        
        Args:
            state: Current 3x3 board state.
            
        Returns:
            Encoded state with shape (3, 3, 3):
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
            encoded_state: Encoded state with shape (3, 3, 3).
            policy: Action probabilities with shape (9,).
            
        Returns:
            List of (encoded_state, policy) tuples for each symmetry.
        """
        # Convert policy from vector to 3x3 grid
        policy_grid = policy.reshape(self.row_count, self.column_count)
        
        # For encoded_state, we need to work with the channels separately
        # Extract the three channels
        channel_0 = encoded_state[0]  # Player 1's pieces
        channel_1 = encoded_state[1]  # Empty spaces
        channel_2 = encoded_state[2]  # Player -1's pieces
        
        symmetries = []
        
        # Generate 4 rotated positions
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