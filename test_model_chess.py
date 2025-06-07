import os
import torch
import numpy as np
import chess
import matplotlib.pyplot as plt
import argparse
from IPython.display import display, HTML

from alphazero.games.chess_game import ChessGame
from alphazero.neural_network.model import ResNet
from alphazero.mcts.search import get_action_distribution


def parse_args():
    parser = argparse.ArgumentParser(description='Test AlphaZero chess model by playing against itself')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--num_simulations', type=int, default=100,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for move selection')
    return parser.parse_args()


def display_board(board):
    """Display the chess board using unicode characters"""
    unicode_pieces = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    }
    
    board_str = str(board)
    for piece, unicode in unicode_pieces.items():
        board_str = board_str.replace(piece, unicode)
    
    # Print the board with row and column labels
    rows = board_str.split('\n')
    print('  a b c d e f g h')
    print(' +-----------------+')
    for i, row in enumerate(rows):
        print(f'{8-i}| {row} |')
    print(' +-----------------+')
    print('  a b c d e f g h')
    print()


def save_board_image(board, move_number, output_dir='chess_game_images'):
    """Save the current board state as an image"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the board
    for i in range(8):
        for j in range(8):
            # Determine square color (light or dark)
            is_light = (i + j) % 2 == 0
            square_color = 'white' if is_light else 'gray'
            
            # Draw the square
            ax.add_patch(plt.Rectangle((j, 7-i), 1, 1, color=square_color))
    
    # Add pieces
    for i in range(8):
        for j in range(8):
            square = chess.square(j, 7-i)
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                color = 'black' if piece_symbol.islower() else 'white'
                piece_type = piece_symbol.upper()
                
                # Map piece type to unicode symbol
                piece_map = {
                    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙'
                }
                if piece_type in piece_map:
                    ax.text(j + 0.5, 7-i + 0.5, piece_map[piece_type], 
                            fontsize=40, ha='center', va='center', color=color)
    
    # Set axis properties
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
    ax.tick_params(length=0)
    ax.set_title(f'Move {move_number}')
    
    # Save the figure
    plt.savefig(f'{output_dir}/move_{move_number:03d}.png')
    plt.close()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize game
    game = ChessGame()
    print(f"Game: {game}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create or load model
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model from {args.checkpoint}")
        model = ResNet.load_checkpoint(args.checkpoint, game=game, device=device)
    else:
        print("Initializing model with random weights")
        model = ResNet(
            game=game,
            input_dim=20,  # Chess has 19 input planes
            num_resblocks=4,  # Smaller model for faster execution
            num_filters=64,
            device=device
        )
    
    # MCTS parameters
    mcts_args = {
        'num_simulations': args.num_simulations,
        'c_puct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25
    }
    
    # Initialize game state
    state = game.get_initial_state()
    current_player = 1  # White starts
    move_number = 0
    
    # Game history for display
    game_history = []
    
    # Play until game is over
    print("\nStarting self-play game...\n")
    print("Initial board:")
    display_board(state)
    save_board_image(state, move_number)
    game_history.append((move_number, None, state.copy()))
    
    while True:
        move_number += 1
        
        # Get canonical state (from current player's perspective)
        canonical_state = game.change_perspective(state, current_player)
        
        # Get action probabilities from MCTS
        print(f"\nMove {move_number}: {'White' if current_player == 1 else 'Black'} to play")
        print("Thinking...")
        
        action_probs, root = get_action_distribution(
            game=game,
            state=canonical_state,
            model=model,
            args=mcts_args,
            temperature=args.temperature,
            add_exploration_noise=True
        )
        print(action_probs[np.argmax(action_probs)])
        # Select action based on probabilities
        valid_moves = game.get_valid_moves(canonical_state)
        
        # Mask invalid moves and renormalize
        masked_probs = action_probs * valid_moves
        sum_probs = np.sum(masked_probs)
        if sum_probs > 0:
            masked_probs /= sum_probs
        else:
            # Fallback to uniform random selection of valid moves
            masked_probs = valid_moves / np.sum(valid_moves)
        
        # Select action based on probabilities
        action = np.random.choice(len(action_probs), p=masked_probs)
        
        # Convert action to UCI move for display
        from alphazero.games.move_encoding import action_int_to_uci
        uci_move = action_int_to_uci(action)
        print(f"Selected move: {uci_move}")
        
        # Execute action
        state = game.get_next_state(state, action, current_player)
        
        # Display the board
        display_board(state)
        save_board_image(state, move_number)
        game_history.append((move_number, uci_move, state.copy()))
        
        # Check if game is over
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            print("\nGame over!")
            if state.is_checkmate():
                winner = "White" if current_player == 1 else "Black"
                print(f"{winner} wins by checkmate!")
            elif state.is_stalemate():
                print("Game drawn by stalemate.")
            elif state.is_insufficient_material():
                print("Game drawn due to insufficient material.")
            elif state.is_fifty_moves():
                print("Game drawn by fifty-move rule.")
            elif state.is_repetition():
                print("Game drawn by threefold repetition.")
            else:
                print("Game drawn.")
            break
        
        # Switch to the other player
        current_player = game.get_opponent(current_player)
    
    # Print game summary
    print("\nGame Summary:")
    print("-" * 40)
    for move_num, move, _ in game_history[1:]:  # Skip initial position
        player = "White" if move_num % 2 == 1 else "Black"
        print(f"Move {move_num}: {player} played {move}")
    
    # Create a GIF of the game
    try:
        from PIL import Image
        import glob
        
        print("\nCreating game animation...")
        images = []
        for filename in sorted(glob.glob('chess_game_images/move_*.png')):
            images.append(Image.open(filename))
        
        # Save as GIF
        images[0].save('chess_game.gif',
                      save_all=True,
                      append_images=images[1:],
                      duration=1000,  # 1 second per frame
                      loop=0)  # Loop forever
        print("Game animation saved as 'chess_game.gif'")
    except ImportError:
        print("PIL library not available. Skipping GIF creation.")
    except Exception as e:
        print(f"Error creating GIF: {e}")


if __name__ == '__main__':
    main()