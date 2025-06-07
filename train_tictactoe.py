import os
import argparse
import torch

from alphazero.games.tictactoe import TicTacToe
from alphazero.neural_network.model import ResNet
from alphazero.trainer import AlphaZero
from alphazero.utils import set_random_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train AlphaZero for Tic Tac Toe')
    
    # Training parameters
    parser.add_argument('--num_iterations', type=int, default=5, 
                        help='Number of training iterations')
    parser.add_argument('--num_selfplay_iterations', type=int, default=20, 
                        help='Number of self-play games per iteration')
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='Number of training epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    
    # Model parameters
    parser.add_argument('--num_resblocks', type=int, default=4, 
                        help='Number of residual blocks in the neural network')
    parser.add_argument('--num_filters', type=int, default=64, 
                        help='Number of filters in convolutional layers')
    
    # MCTS parameters
    parser.add_argument('--num_simulations', type=int, default=50, 
                        help='Number of MCTS simulations per move')
    parser.add_argument('--c_puct', type=float, default=1.0, 
                        help='Exploration constant in PUCT formula')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, 
                        help='Alpha parameter for Dirichlet noise')
    parser.add_argument('--dirichlet_epsilon', type=float, default=0.25, 
                        help='Epsilon parameter for Dirichlet noise')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='models', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default='',
                        help='Load checkpoint of previously trained models')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Directory to save logs')
    
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
    game = TicTacToe()
    print(f"Game: {game}")
    
    # Initialize neural network model
    model = ResNet(
        game=game,
        num_resblocks=args.num_resblocks,
        num_filters=args.num_filters,
        device=device
    )
    if args.load_checkpoint != '':
        model.load_checkpoint(args.load_checkpoint, game)
    print(f"Model: {model.__class__.__name__} with {args.num_resblocks} residual blocks")
    
    # Prepare training arguments
    training_args = {
        'num_iterations': args.num_iterations,
        'num_selfplay_iterations': args.num_selfplay_iterations,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': device,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'max_examples': 100000,  # Limit memory usage
        
        # MCTS parameters
        'num_simulations': args.num_simulations,
        'c_puct': args.c_puct,
        'dirichlet_alpha': args.dirichlet_alpha,
        'dirichlet_epsilon': args.dirichlet_epsilon,
        
        # Temperature schedule
        'temperature_threshold': 10,  # Move number to decrease temperature
    }
    
    # Initialize AlphaZero trainer
    alpha_zero = AlphaZero(
        game=game,
        model=model,
        args=training_args
    )
    
    # Train the model
    trained_model = alpha_zero.train()
    
    # Save the final model
    final_model_path = os.path.join(args.checkpoint_dir, 'tictactoe_final_model.pt')
    trained_model.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main() 