import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from alphazero.games.tictactoe import TicTacToe
from alphazero.neural_network.model import ResNet
from alphazero.trainer import AlphaZero
from alphazero.utils import set_random_seed, visualize_board


def main():
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Define paths
    model_dir = 'models'
    log_dir = 'logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, 'test.pt')
    
    # Initialize game
    game = TicTacToe()
    print(f"Game: {game}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model with minimal parameters for quick training
    model = ResNet(
        game=game,
        num_resblocks=2,  # Fewer resblocks for faster training
        num_filters=32,   # Fewer filters for faster training
        device=device
    )
    
    # Training arguments (minimal for quick test)
    training_args = {
        'num_iterations': 2,             # Just 2 iterations
        'num_selfplay_iterations': 5,    # 5 self-play games per iteration
        'num_epochs': 5,                 # 5 training epochs
        'batch_size': 32,                # Small batch size
        'lr': 0.001,                     # Learning rate
        'device': device,
        'checkpoint_dir': model_dir,
        'log_dir': log_dir,
        
        # MCTS parameters (reduced for speed)
        'num_simulations': 25,           # Few simulations per move
        'c_puct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        
        # Temperature schedule
        'temperature_threshold': 5,      # Move number to decrease temperature
    }
    
    print("Starting quick training (this will take a few minutes)...")
    start_time = time.time()
    
    # Initialize AlphaZero trainer
    alpha_zero = AlphaZero(
        game=game,
        model=model,
        args=training_args
    )
    
    # Train the model
    trained_model = alpha_zero.train()
    
    # Save the final model
    trained_model.save_checkpoint(final_model_path)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Model saved to {final_model_path}")
    
    # TEST THE MODEL ON THE SAME STATE AS IN THE NOTEBOOK
    print("\n======= TESTING MODEL =======")
    
    # Create the same state as in the notebook example
    state = game.get_initial_state()
    state = game.get_next_state(state, 2, -1)  # -1 at position 2
    state = game.get_next_state(state, 4, -1)  # -1 at position 4 
    state = game.get_next_state(state, 6, 1)   # 1 at position 6
    state = game.get_next_state(state, 8, 1)   # 1 at position 8
    
    print("Board state:")
    print(state)
    
    # Visualize the board
    visualize_board(state)
    
    # Encode the state for the neural network
    encoded_state = game.get_encoded_state(state)
    tensor_state = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
    
    # Use the trained model
    trained_model.eval()
    
    # Get predictions
    with torch.no_grad():
        policy, value = trained_model(tensor_state.to(device))
        value = value.item()
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
    
    print(f"Predicted value: {value:.4f}")
    
    # Get valid moves
    valid_moves = game.get_valid_moves(state)
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
    bars = plt.bar(range(game.action_size), policy)
    
    # Highlight valid moves
    for i, bar in enumerate(bars):
        if valid_moves[i] == 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('AlphaZero Policy Distribution')
    plt.xticks(range(game.action_size))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Run MCTS search to get improved policy
    from alphazero.mcts.search import get_action_distribution
    
    mcts_args = {
        'num_simulations': 100,  # More simulations for better results
        'c_puct': 1.0,
    }
    
    # Current player's perspective (next player is X/1)
    canonical_state = game.change_perspective(state, 1)
    
    print("\nRunning MCTS search...")
    mcts_policy, _ = get_action_distribution(
        game, canonical_state, trained_model, mcts_args,
        temperature=0.1, add_exploration_noise=False
    )
    
    # Plot MCTS policy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(game.action_size), mcts_policy)
    
    # Highlight valid moves and winning move
    for i, bar in enumerate(bars):
        if valid_moves[i] == 1:
            # Check if this move would be a winning move
            test_state = state.copy()
            test_state = game.get_next_state(test_state, i, 1)
            if game.check_win(test_state, i):
                bar.set_color('gold')  # Highlight winning move
            else:
                bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Find best move
    best_move = np.argmax(mcts_policy)
    print(f"Best move according to MCTS: {best_move}")
    
    # Check if the best move is a winning move
    test_state = state.copy()
    test_state = game.get_next_state(test_state, best_move, 1)
    if game.check_win(test_state, best_move):
        print("This is a winning move!")
    else:
        print("This is not a winning move.")
    
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('AlphaZero MCTS Policy Distribution')
    plt.xticks(range(game.action_size))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main() 