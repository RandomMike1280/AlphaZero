from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):
    """
    Abstract base class for game implementations to be used with AlphaZero.
    
    Any game implementation must inherit from this class and implement all
    the abstract methods.
    """
    
    def __init__(self):
        """Initialize the game parameters."""
        pass
    
    def __repr__(self):
        """String representation of the game."""
        return self.__class__.__name__
    
    @abstractmethod
    def get_initial_state(self):
        """
        Returns the initial state of the game.
        
        Returns:
            Initial state (typically a numpy array).
        """
        pass
    
    @abstractmethod
    def get_next_state(self, state, action, player):
        """
        Returns the state that results from the given action.
        
        Args:
            state: Current state.
            action: Action to take.
            player: Current player (1 or -1).
            
        Returns:
            Next state after the action is applied.
        """
        pass
    
    @abstractmethod
    def get_valid_moves(self, state):
        """
        Returns a binary vector of valid moves for the given state.
        
        Args:
            state: Current state.
            
        Returns:
            Binary vector of valid moves (1 = valid, 0 = invalid).
        """
        pass
    
    @abstractmethod
    def check_win(self, state, action):
        """
        Checks if the last action led to a win.
        
        Args:
            state: Current state.
            action: Last action taken.
            
        Returns:
            True if the last action led to a win, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_value_and_terminated(self, state, action):
        """
        Checks if the game is over and returns the value of the state.
        
        Args:
            state: Current state.
            action: Last action taken.
            
        Returns:
            (value, terminated): Value is from the perspective of the player who
            just moved. terminated is True if the game is over.
        """
        pass
    
    def get_opponent(self, player):
        """
        Returns the opponent of the given player.
        
        Args:
            player: Current player (1 or -1).
            
        Returns:
            Opponent player (-1 or 1).
        """
        return -player
    
    def get_opponent_value(self, value):
        """
        Returns the value from the opponent's perspective.
        
        Args:
            value: Value from current player's perspective.
            
        Returns:
            Value from opponent's perspective.
        """
        return -value
    
    @abstractmethod
    def change_perspective(self, state, player):
        """
        Changes the perspective of the state to the given player.
        
        Args:
            state: Current state.
            player: Player to change perspective to (1 or -1).
            
        Returns:
            State from the perspective of the given player.
        """
        pass
    
    @abstractmethod
    def get_encoded_state(self, state):
        """
        Encodes the state for neural network input.
        
        Args:
            state: Current state.
            
        Returns:
            Encoded state suitable for neural network input.
        """
        pass 