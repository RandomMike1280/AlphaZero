import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from alphazero.games.tictactoe import TicTacToe
from alphazero.utils import visualize_board

def main():
    # Initialize game
    game = TicTacToe()
    print(f"Game: {game}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a simple board state
    state = game.get_initial_state()
    state = game.get_next_state(state, 2, -1)  # -1 at position 2
    state = game.get_next_state(state, 4, -1)  # -1 at position 4 
    state = game.get_next_state(state, 6, 1)   # 1 at position 6
    state = game.get_next_state(state, 8, 1)   # 1 at position 8
    
    print("Board state:")
    print(state)
    
    # Get valid moves
    valid_moves = game.get_valid_moves(state)
    print("Valid moves:", np.where(valid_moves == 1)[0].tolist())
    
    # Use the visualize_board function from alphazero.utils
    visualize_board(state, game_type='tictactoe')

if __name__ == '__main__':
    main() 