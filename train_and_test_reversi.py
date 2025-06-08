import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from alphazero.games.reversi import Reversi
from alphazero.neural_network.model import ResNet
from alphazero.trainer import AlphaZero
from alphazero.mcts.search import get_action_distribution
from alphazero.utils import set_random_seed


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load the weights into.
        optimizer: Optional optimizer to load state.
        
    Returns:
        Tuple of (model, optimizer, start_iteration, best_score)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_iteration = checkpoint.get('iteration', 0) + 1
    best_score = checkpoint.get('best_score', float('-inf'))
    
    print(f"Loaded checkpoint from iteration {start_iteration-1}")
    return model, optimizer, start_iteration, best_score


def train_model(args, checkpoint_path=None):
    """
    Train an AlphaZero model for Reversi.
    
    Args:
        args: Dictionary of training arguments.
        checkpoint_path: Optional path to checkpoint file to resume training.
    
    Returns:
        Trained model.
    """
    print("\n=== TRAINING MODEL ===\n")
    
    # Initialize game and model
    game = Reversi()
    print(f"Game: {game}")
    
    model = ResNet(
        game=game,
        num_resblocks=args['num_resblocks'],
        num_filters=args['num_filters'],
        device=args['device']
    )
    
    start_iteration = 0
    
    # Load checkpoint if provided
    if checkpoint_path:
        try:
            model, _, start_iteration, _ = load_checkpoint(checkpoint_path, model)
            print(f"Resuming training from iteration {start_iteration}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    # Create trainer
    alpha_zero = AlphaZero(
        game=game,
        model=model,
        args=args
    )
    
    # Train model
    start_time = time.time()
    trained_model = alpha_zero.train(start_iteration=start_iteration)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Model saved to {args['final_model_path']}")
    
    return trained_model


def test_model(model, num_games=10):
    """
    Test the trained model by playing against a random player.
    
    Args:
        model: Trained AlphaZero model.
        num_games: Number of games to play.
    """
    print("\n=== TESTING MODEL ===\n")
    
    game = Reversi()
    device = next(model.parameters()).device
    
    results = []
    
    for game_num in range(1, num_games + 1):
        state = game.get_initial_state()
        player = 1  # Model is player 1 (Black)
        
        while True:
            if player == 1:  # Model's turn
                # Get model's action
                encoded_state = game.get_encoded_state(state)
                encoded_state = torch.tensor(encoded_state, device=device).unsqueeze(0)
                
                # Get action probabilities from MCTS
                action_probs, _ = get_action_distribution(model, encoded_state, game, 1.0, 100)
                
                # Choose action with highest probability
                action = np.argmax(action_probs)
                
                # If the chosen action is invalid, choose a random valid move
                valid_moves = game.get_valid_moves(state)
                if valid_moves[action] == 0:
                    valid_indices = np.where(valid_moves == 1)[0]
                    if len(valid_indices) > 0:
                        action = np.random.choice(valid_indices)
                    else:  # No valid moves, pass
                        action = None
            else:  # Random player's turn
                valid_moves = game.get_valid_moves(-state)  # Invert state for player -1
                valid_indices = np.where(valid_moves == 1)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:  # No valid moves, pass
                    action = None
            
            # Apply the action
            if player == 1:
                state = game.get_next_state(state, action, 1)
            else:
                state = game.get_next_state(-state, action, -1)
            
            # Check if game is over
            winner, terminated = game.get_value_and_terminated(state, action)
            if terminated:
                results.append(winner)
                print(f"Game {game_num}: {'Model (Black) wins!' if winner == 1 else 'Random (White) wins!' if winner == -1 else 'Draw!'}")
                break
                
            # Switch player
            player *= -1
    
    # Print summary
    model_wins = sum(1 for r in results if r == 1)
    random_wins = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    
    print("\n=== TEST RESULTS ===")
    print(f"Model (Black) wins: {model_wins}/{num_games} ({model_wins/num_games*100:.1f}%)")
    print(f"Random (White) wins: {random_wins}/{num_games} ({random_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)")


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and test AlphaZero for Reversi')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint file to resume training')
    parser.add_argument('--test-only', action='store_true',
                      help='Only test the model without training')
    args_cmd = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Training arguments
    args = {
        'num_iterations': 3,  # Number of training iterations
        'num_selfplay_iterations': 10,  # Number of self-play games per iteration
        'mcts_batch_size':32,
        'num_epochs': 4,  # Number of training epochs per iteration
        'batch_size': 64,  # Batch size for training
        'learning_rate': 0.001,  # Learning rate
        'num_mcts_simulations': 100,  # Number of MCTS simulations per move
        'num_resblocks': 5,  # Number of residual blocks in the neural network
        'num_filters': 64,  # Number of filters in the neural network
        'c_puct': 1.0,  # Exploration constant for MCTS
        'temp_threshold': 15,  # Temperature threshold for exploration
        'dirichlet_alpha': 0.3,  # Dirichlet noise alpha
        'dirichlet_epsilon': 0.25,  # Dirichlet noise epsilon
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Device to use for training
        'checkpoint_dir': 'checkpoints/reversi',  # Directory to save checkpoints
        'final_model_path': 'models/reversi_model.pt',  # Path to save the final model
    }
    
    # Create directories if they don't exist
    os.makedirs(args['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(args['final_model_path']), exist_ok=True)
    
    print(f"Using device: {args['device']}")
    
    # Train and test the model
    if not args_cmd.test_only:
        trained_model = train_model(args, checkpoint_path=args_cmd.checkpoint)
    else:
        if not args_cmd.checkpoint:
            raise ValueError("Checkpoint path must be provided when using --test-only")
        game = Reversi()
        trained_model = ResNet(
            game=game,
            num_resblocks=args['num_resblocks'],
            num_filters=args['num_filters'],
            device=args['device']
        )
        trained_model, _, _, _ = load_checkpoint(args_cmd.checkpoint, trained_model)
    
    test_model(trained_model, num_games=10)
