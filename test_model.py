import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse

from alphazero.games.tictactoe import TicTacToe
from alphazero.neural_network.model import ResNet
from alphazero.utils import visualize_board


def parse_args():
    parser = argparse.ArgumentParser(description='Test AlphaZero model on specific board state')
    parser.add_argument('--model_path', type=str, default='models/tictactoe_final_model.pt',
                        help='Path to the trained model checkpoint')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train_tictactoe.py or specify correct path.")
        return
    
    # Initialize game
    tictactoe = TicTacToe()
    
    # Create the same state as in the notebook example
    state = tictactoe.get_initial_state()
    state = tictactoe.get_next_state(state, 2, -1)  # -1 at position 2
    # state = game.get_next_state(state, 3, -1)  # -1 at position 4 
    # state = game.get_next_state(state, 6, 1)   # 1 at position 6
    state = tictactoe.get_next_state(state, 0, 1)   # 1 at position 8
    
    print("Board state:")
    print(state)
    
    # Visualize the board
    visualize_board(state)
    
    # Encode the state for the neural network
    encoded_state = tictactoe.get_encoded_state(state)
    tensor_state = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
    
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet.load_checkpoint(args.model_path, tictactoe, device)
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        policy, value = model(tensor_state.to(device))
        value = value.item()
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
    
    print(f"Predicted value: {value:.4f}")
    
    # Get valid moves
    valid_moves = tictactoe.get_valid_moves(state)
    print("Valid moves:", np.where(valid_moves == 1)[0].tolist())
    
    # Get the top 3 moves with their probabilities
    valid_policy = policy * valid_moves
    valid_policy = valid_policy / np.sum(valid_policy) if np.sum(valid_policy) > 0 else valid_policy
    
    top_moves = np.argsort(valid_policy)[-3:][::-1]
    print("\nTop 3 moves with probabilities:")
    for move in top_moves:
        if valid_moves[move] == 1:
            print(f"Move {move}: {valid_policy[move]:.4f}")
    
    # Plot the policy distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(tictactoe.action_size), policy)
    
    # Highlight valid moves
    for i, bar in enumerate(bars):
        if valid_moves[i] == 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('AlphaZero Policy Distribution')
    plt.xticks(range(tictactoe.action_size))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Run MCTS search to get improved policy
    from alphazero.mcts.search import get_action_distribution
    
    mcts_args = {
        'num_simulations': 1000,
        'c_puct': 1.0
    }
    
    # Current player's perspective (next player is X/1)
    canonical_state = tictactoe.change_perspective(state, 1)
    
    print("\nRunning MCTS search...")
    mcts_policy, _ = get_action_distribution(
        tictactoe, canonical_state, model, mcts_args,
        temperature=0.1, add_exploration_noise=False
    )
    
    # Plot MCTS policy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(tictactoe.action_size), mcts_policy)
    
    # Highlight valid moves
    for i, bar in enumerate(bars):
        if valid_moves[i] == 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Find best move
    best_move = np.argmax(mcts_policy)
    print(f"Best move according to MCTS: {best_move}")
    top_moves = np.argsort(mcts_policy)[-3:][::-1]
    print("\nTop 3 moves with probabilities:")
    for move in top_moves:
        if valid_moves[move] == 1:
            print(f"Move {move}: {mcts_policy[move]:.7f}")
    
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('AlphaZero MCTS Policy Distribution')
    plt.xticks(range(tictactoe.action_size))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main() 