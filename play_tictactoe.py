import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from alphazero.games.tictactoe import TicTacToe
from alphazero.neural_network.model import ResNet
from alphazero.utils import play_move, visualize_board


def parse_args():
    parser = argparse.ArgumentParser(description='Play against AlphaZero Tic Tac Toe')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--num_simulations', type=int, default=100,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--human_player', type=int, default=1, choices=[-1, 1],
                        help='Player number for human (1 or -1)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for move selection')
    
    return parser.parse_args()


def get_human_move(state):
    """
    Get a move from the human player.
    
    Args:
        state: Current game state.
        
    Returns:
        The selected action.
    """
    valid_moves = (state.reshape(-1) == 0).astype(np.uint8)
    
    # Display the board with cell numbers
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros((3, 3)), cmap='binary', alpha=0.1, extent=[0, 3, 0, 3])
    
    # Draw grid lines
    for i in range(4):
        plt.axhline(i, color='black', lw=2)
        plt.axvline(i, color='black', lw=2)
    
    # Mark positions and show cell numbers
    for i in range(3):
        for j in range(3):
            cell_num = i * 3 + j
            if state[i, j] == 1:
                plt.text(j + 0.5, i + 0.5, 'X', fontsize=40, ha='center', va='center')
            elif state[i, j] == -1:
                plt.text(j + 0.5, i + 0.5, 'O', fontsize=40, ha='center', va='center')
            else:
                plt.text(j + 0.5, i + 0.5, str(cell_num), fontsize=20, ha='center', va='center')
    
    plt.title('Enter the number of your move (0-8)')
    plt.tight_layout()
    plt.show()
    
    # Get input from user
    while True:
        try:
            action = int(input("Enter your move (0-8): "))
            if 0 <= action <= 8 and valid_moves[action] == 1:
                return action
            else:
                print("Invalid move. Please choose an empty cell.")
        except ValueError:
            print("Please enter a number between 0 and 8.")


def display_result(state, action, game, player):
    """
    Display the final result of the game.
    
    Args:
        state: Final game state.
        action: Last action taken.
        game: Game instance.
        player: Last player to move.
    """
    visualize_board(state)
    
    value, _ = game.get_value_and_terminated(state, action)
    
    if value == 1:
        if player == 1:
            print("Game over: X wins!")
        else:
            print("Game over: O wins!")
    else:
        print("Game over: It's a draw!")


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize game
    game = TicTacToe()
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet.load_checkpoint(args.model_path, game, device)
    print(f"Loaded model from {args.model_path}")
    
    # Initialize game state
    state = game.get_initial_state()
    
    # Track game state
    current_player = 1  # X goes first
    action_history = []
    
    # Game loop
    while True:
        print(f"Current player: {'X' if current_player == 1 else 'O'}")
        
        if current_player == args.human_player:
            # Human's turn
            action = get_human_move(state)
        else:
            # AlphaZero's turn
            from alphazero.mcts.search import get_action_distribution
            
            mcts_args = {
                'num_simulations': args.num_simulations,
                'c_puct': 1.0,
            }
            
            # Get canonical state (from AI's perspective)
            canonical_state = game.change_perspective(state, current_player)
            
            # Get action distribution
            action_probs, _ = get_action_distribution(
                game, canonical_state, model, mcts_args, 
                temperature=args.temperature, add_exploration_noise=False
            )
            
            # Select move with highest probability
            action = np.argmax(action_probs)
            
            print(f"AlphaZero chose move: {action}")
        
        # Apply the move
        state = game.get_next_state(state, action, current_player)
        action_history.append(action)
        
        # Clear previous visualizations
        plt.close('all')
        
        # Check if the game is over
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            display_result(state, action, game, current_player)
            
            # Ask to play again
            play_again = input("Play again? (y/n): ").lower()
            if play_again != 'y':
                break
                
            # Reset the game
            state = game.get_initial_state()
            current_player = 1
            action_history = []
        else:
            # Visualize the current board
            visualize_board(state)
            
            # Switch to the other player
            current_player = game.get_opponent(current_player)


if __name__ == '__main__':
    # Add torch import here to avoid circular imports
    import torch
    main() 