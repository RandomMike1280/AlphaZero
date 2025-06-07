import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from alphazero.games.tictactoe import TicTacToe
from alphazero.neural_network.model import ResNet
from alphazero.mcts.search import MCTS, get_action_distribution


def main():
    # Initialize game
    game = TicTacToe()
    print(f"Game: {game}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a minimal model for testing
    model = ResNet.load_checkpoint("models/tictactoe_final_model.pt", game, device)
    print("Model created")
    
    # Create a simple board state
    state = game.get_initial_state()
    state = game.get_next_state(state, 2, -1)  # -1 at position 2
    # state = game.get_next_state(state, 3, -1)  # -1 at position 4 
    # state = game.get_next_state(state, 6, 1)   # 1 at position 6
    state = game.get_next_state(state, 0, 1)   # 1 at position 8
    
    print("Board state:")
    print(state)
    
    # Visualize the board as a 3x3 grid
    plt.figure(figsize=(5, 5))
    plt.imshow(np.zeros((3, 3)), cmap='binary', alpha=0.1)
    for i in range(3):
        for j in range(3):
            if state[i, j] == 1:  # X
                plt.text(j, i, 'X', fontsize=30, ha='center', va='center')
            elif state[i, j] == -1:  # O
                plt.text(j, i, 'O', fontsize=30, ha='center', va='center')
    plt.grid(True)
    plt.xticks([0, 1, 2])
    plt.yticks([0, 1, 2])
    plt.title('Board State')
    plt.savefig('board_state.png')
    print("Board visualization saved to board_state.png")
    
    # Get valid moves
    valid_moves = game.get_valid_moves(state)
    print("Valid moves:", np.where(valid_moves == 1)[0].tolist())
    
    # Encode the state for the neural network
    encoded_state = game.get_encoded_state(state)
    print(f"Encoded state shape: {encoded_state.shape}")
    
    # Use the model to predict
    tensor_state = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
    model.eval()
    
    with torch.no_grad():
        policy, value = model(tensor_state.to(device))
        value = value.item()
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
    
    print(f"Raw model prediction - Value: {value:.4f}")
    
    # Get top 3 moves from raw policy
    valid_policy = policy * valid_moves
    valid_policy_normalized = valid_policy / np.sum(valid_policy) if np.sum(valid_policy) > 0 else valid_policy
    # valid_policy_normalized = valid_policy / np.sum(valid_policy) if np.sum(valid_policy) > 0 else valid_policy
    top_indices = np.argsort(valid_policy_normalized)[-3:][::-1]
    
    print("Top 3 moves from raw policy:")
    for i, idx in enumerate(top_indices):
        row, col = idx // 3, idx % 3
        print(f"  {i+1}. Position ({row},{col}) [idx={idx}]: {valid_policy_normalized[idx]:.4f}")
    
    # Use MCTS to get an improved policy
    args = {
        'num_simulations': 100,  # More simulations for better results
        'c_puct': 1.0,
    }
    
    # Current player's perspective (next player is X/1)
    canonical_state = game.change_perspective(state, 1)
    
    print("\nRunning MCTS search...")
    try:
        mcts_policy, root_node = get_action_distribution(
            game, canonical_state, model, args,
            temperature=0.1, add_exploration_noise=False
        )
        # print(f"MCTS search completed. Value: {root_node.value:.4f}")
        
        # Get top 3 moves from MCTS policy
        valid_mcts_policy = mcts_policy * valid_moves
        top_mcts_indices = np.argsort(valid_mcts_policy)[-3:][::-1]
        
        print("Top 3 moves from MCTS policy:")
        for i, idx in enumerate(top_mcts_indices):
            row, col = idx // 3, idx % 3
            print(f"  {i+1}. Position ({row},{col}) [idx={idx}]: {valid_mcts_policy[idx]:.4f}")
        
        # Display MCTS policy
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(game.action_size), mcts_policy)
        
        # Highlight valid moves and winning move if any
        for i, bar in enumerate(bars):
            if valid_moves[i] == 1:
                # Check if this move would be a winning move
                test_state = state.copy()
                test_state = game.get_next_state(test_state, i, 1)
                if game.check_win(test_state, i):
                    bar.set_color('gold')  # Highlight winning move
                    print(f"Move {i} at position ({i//3},{i%3}) is a winning move!")
                else:
                    bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.xlabel('Action')
        plt.ylabel('Probability')
        plt.title('MCTS Policy Distribution')
        plt.xticks(range(game.action_size))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('mcts_policy.png')
        print("MCTS policy visualization saved to mcts_policy.png")
        
    except Exception as e:
        print(f"Error in MCTS: {e}")
    
    # Display raw policy distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(game.action_size), policy)
    
    # Highlight valid moves
    for i, bar in enumerate(bars):
        if valid_moves[i] == 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('Raw Policy Distribution')
    plt.xticks(range(game.action_size))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('raw_policy.png')
    print("Raw policy visualization saved to raw_policy.png")


if __name__ == '__main__':
    main() 