import os
import argparse
import torch

from alphazero.games.chess_game import ChessGame
from alphazero.neural_network.model import ResNet
from alphazero.trainer import AlphaZero
from alphazero.utils import set_random_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train AlphaZero for Chess')
    
    # Training parameters
    parser.add_argument('--num_iterations', type=int, default=25, 
                        help='Number of training iterations')
    parser.add_argument('--num_selfplay_iterations', type=int, default=100, 
                        help='Number of self-play games per iteration')
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='Number of training epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay for regularization')
    
    # Model parameters
    parser.add_argument('--num_resblocks', type=int, default=19, 
                        help='Number of residual blocks in the neural network')
    parser.add_argument('--num_filters', type=int, default=256, 
                        help='Number of filters in convolutional layers')
    
    # MCTS parameters
    parser.add_argument('--num_simulations', type=int, default=200, 
                        help='Number of MCTS simulations per move')
    parser.add_argument('--c_puct', type=float, default=1.0, 
                        help='Exploration constant in PUCT formula')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, 
                        help='Alpha parameter for Dirichlet noise')
    parser.add_argument('--dirichlet_epsilon', type=float, default=0.25, 
                        help='Epsilon parameter for Dirichlet noise')
    
    # Self-play parameters
    parser.add_argument('--temperature_threshold', type=int, default=20, 
                        help='Move number to decrease temperature')
    parser.add_argument('--max_moves', type=int, default=512, 
                        help='Maximum number of moves per game')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='models/chess', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/chess', 
                        help='Directory to save logs')
    parser.add_argument('--resume_from', type=int, default=None,
                        help='Resume training from a specific iteration')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Get device (CPU or GPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize game
    game = ChessGame()
    print(f"Game: {game}")
    
    # Initialize neural network model
    model = ResNet(
        game=game,
        input_dim=19,  # Input dimension for chess (19 planes)
        num_resblocks=args.num_resblocks,
        num_filters=args.num_filters,
        device=device
    )
    print(f"Model: {model.__class__.__name__} with {args.num_resblocks} residual blocks")
    
    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Prepare training arguments
    training_args = {
        'num_iterations': args.num_iterations,
        'num_selfplay_iterations': args.num_selfplay_iterations,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'device': device,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'max_examples': 500000,  # Limit memory usage
        
        # MCTS parameters
        'num_simulations': args.num_simulations,
        'c_puct': args.c_puct,
        'dirichlet_alpha': args.dirichlet_alpha,
        'dirichlet_epsilon': args.dirichlet_epsilon,
        
        # Self-play parameters
        'temperature_threshold': args.temperature_threshold,
        'max_moves': args.max_moves,
    }
    
    # Initialize AlphaZero trainer
    alpha_zero = AlphaZero(
        game=game,
        model=model,
        args=training_args
    )
    
    # Resume training if specified
    if args.resume_from is not None:
        print(f"Resuming training from iteration {args.resume_from}")
        success = alpha_zero.load_training_state(args.resume_from)
        if not success:
            print("Failed to load training state. Starting from scratch.")
    
    # Train the model
    trained_model = alpha_zero.train()
    
    # Save the final model
    final_model_path = os.path.join(args.checkpoint_dir, 'chess_final_model.pt')
    trained_model.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main() 