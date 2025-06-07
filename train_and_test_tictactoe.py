import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from alphazero.games.tictactoe import TicTacToe
from alphazero.neural_network.model import ResNet
from alphazero.trainer import AlphaZero
from alphazero.mcts.search import get_action_distribution
from alphazero.utils import set_random_seed


def train_model(args):
    """
    Train an AlphaZero model for Tic Tac Toe.
    
    Args:
        args: Dictionary of training arguments.
    
    Returns:
        Trained model.
    """
    print("\n=== TRAINING MODEL ===\n")
    
    # Initialize game and model
    game = TicTacToe()
    print(f"Game: {game}")
    
    model = ResNet(
        game=game,
        num_resblocks=args['num_resblocks'],
        num_filters=args['num_filters'],
        device=args['device']
    )
    
    # Create trainer
    alpha_zero = AlphaZero(
        game=game,
        model=model,
        args=args
    )
    
    # Train model
    start_time = time.time()
    trained_model = alpha_zero.train()
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Model saved to {args['final_model_path']}")
    
    return trained_model


def test_model(model, specific_state=True):
    """
    Test the trained model on a specific board state or initial state.
    
    Args:
        model: Trained AlphaZero model.
        specific_state: Whether to test on a specific board state or initial state.
    """
    print("\n=== TESTING MODEL ===\n")
    
    game = TicTacToe()
    
    # Create board state
    if specific_state:
        # Create the same state as in the notebook example
        state = game.get_initial_state()
        state = game.get_next_state(state, 2, -1)  # -1 at position 2
        state = game.get_next_state(state, 4, -1)  # -1 at position 4 
        state = game.get_next_state(state, 6, 1)   # 1 at position 6
        state = game.get_next_state(state, 8, 1)   # 1 at position 8
        print("Testing on specific board state:")
    else:
        # Create a random state
        state = game.get_initial_state()
        print("Testing on initial board state:")
    
    print(state)
    
    # Visualize the board
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
    plt.savefig('test_board_state.png')
    print("Board visualization saved to test_board_state.png")
    
    # Get valid moves
    valid_moves = game.get_valid_moves(state)
    print("Valid moves:", np.where(valid_moves == 1)[0].tolist())
    
    # Encode the state for the neural network
    encoded_state = game.get_encoded_state(state)
    tensor_state = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
    
    # Use the model to predict
    model.eval()
    with torch.no_grad():
        policy, value = model(tensor_state.to(args['device']))
        value = value.item()
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
    
    print(f"Model prediction - Value: {value:.4f}")
    
    # Get top moves from raw policy
    valid_policy = policy * valid_moves
    valid_policy_normalized = valid_policy / np.sum(valid_policy) if np.sum(valid_policy) > 0 else valid_policy
    top_indices = np.argsort(valid_policy_normalized)[-3:][::-1]
    
    print("Top 3 moves from raw policy:")
    for i, idx in enumerate(top_indices):
        row, col = idx // 3, idx % 3
        print(f"  {i+1}. Position ({row},{col}) [idx={idx}]: {valid_policy_normalized[idx]:.4f}")
    
    # Plot the raw policy distribution
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
    plt.savefig('test_raw_policy.png')
    print("Raw policy visualization saved to test_raw_policy.png")
    
    # Run MCTS to get improved policy
    mcts_args = {
        'num_simulations': 400,  # More simulations for better results
        'c_puct': 1.0,
    }
    
    # Current player's perspective (next player is X/1)
    canonical_state = game.change_perspective(state, 1)
    
    print("\nRunning MCTS search...")
    mcts_policy, _ = get_action_distribution(
        game, canonical_state, model, mcts_args,
        temperature=0.1, add_exploration_noise=False
    )
    
    # Get top moves from MCTS policy
    valid_mcts_policy = mcts_policy * valid_moves
    valid_mcts_normalized = valid_mcts_policy / np.sum(valid_mcts_policy) if np.sum(valid_mcts_policy) > 0 else valid_mcts_policy
    top_mcts_indices = np.argsort(valid_mcts_normalized)[-3:][::-1]
    
    print("Top 3 moves from MCTS policy:")
    for i, idx in enumerate(top_mcts_indices):
        row, col = idx // 3, idx % 3
        print(f"  {i+1}. Position ({row},{col}) [idx={idx}]: {valid_mcts_normalized[idx]:.4f}")
    
    # Check for winning moves
    winning_moves = []
    for i in range(game.action_size):
        if valid_moves[i] == 1:
            test_state = state.copy()
            test_state = game.get_next_state(test_state, i, 1)
            if game.check_win(test_state, i):
                winning_moves.append(i)
    
    if winning_moves:
        print("Winning moves found:", winning_moves)
        print(f"MCTS selection of winning move: {np.argmax(mcts_policy) in winning_moves}")
    else:
        print("No winning moves available.")
    
    # Plot MCTS policy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(game.action_size), mcts_policy)
    
    # Highlight valid moves and winning moves
    for i, bar in enumerate(bars):
        if i in winning_moves:
            bar.set_color('gold')  # Winning move
        elif valid_moves[i] == 1:
            bar.set_color('green')  # Valid non-winning move
        else:
            bar.set_color('red')  # Invalid move
    
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('MCTS Policy Distribution')
    plt.xticks(range(game.action_size))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_mcts_policy.png')
    print("MCTS policy visualization saved to test_mcts_policy.png")


if __name__ == '__main__':
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Define paths
    model_dir = 'models'
    log_dir = 'logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, 'tictactoe_quick_train.pt')
    
    # Training parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    args = {
        # Model parameters
        'num_resblocks': 4,         # Small model for quick training
        'num_filters': 64,          # Fewer filters for faster training
        'device': device,
        
        # Training parameters
        'num_iterations': 3,        # Just 3 iterations
        'num_selfplay_iterations': 10,  # 10 self-play games per iteration
        'num_epochs': 10,           # 10 training epochs
        'batch_size': 64,           # Small batch size
        'lr': 0.001,                # Learning rate
        'checkpoint_dir': model_dir,
        'log_dir': log_dir,
        'final_model_path': final_model_path,
        
        # MCTS parameters
        'num_simulations': 50,      # Few simulations per move for speed
        'c_puct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        
        # Temperature schedule
        'temperature_threshold': 10,  # Move number to decrease temperature
    }
    
    # Train the model
    trained_model = train_model(args)
    
    # Test the model on the specific board state
    test_model(trained_model, specific_state=True) 