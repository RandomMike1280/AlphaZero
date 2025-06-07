import os
import argparse
import numpy as np
import chess
import chess.svg
import cairosvg
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch

from alphazero.games.chess_game import ChessGame
from alphazero.neural_network.model import ResNet
from alphazero.mcts.search import get_action_distribution


def parse_args():
    parser = argparse.ArgumentParser(description='Play against AlphaZero Chess')
    
    parser.add_argument('--model_path', type=str, default='None',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--num_simulations', type=int, default=800,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--human_player', type=str, default='white', choices=['white', 'black'],
                        help='Player color for human ("white" or "black")')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for move selection')
    
    return parser.parse_args()


def display_board(board):
    """
    Display the chess board using matplotlib.
    
    Args:
        board: python-chess Board object.
    """
    # Generate SVG
    svg_str = chess.svg.board(board, size=400)
    
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
    
    # Display the image
    img = Image.open(io.BytesIO(png_data))
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.show()


def get_human_move(board):
    """
    Get a move from the human player.
    
    Args:
        board: python-chess Board object.
        
    Returns:
        The selected move.
    """
    display_board(board)
    
    # List legal moves for reference
    print("\nLegal moves:")
    for i, move in enumerate(board.legal_moves):
        print(f"{i+1}: {move.uci()}", end=" ")
        if (i+1) % 5 == 0:
            print()
    print("\n")
    
    # Get input from user
    while True:
        try:
            move_str = input("Enter your move (e.g. 'e2e4' or 'g1f3'): ")
            
            # Convert to chess.Move
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print("Invalid format. Use format like 'e2e4'.")
                continue
                
            if move in board.legal_moves:
                return move
            else:
                print("Invalid move. Please choose a legal move.")
        except Exception as e:
            print(f"Error: {e}")
            print("Please enter a valid move.")


def get_ai_move(board, model, game, num_simulations, temperature):
    """
    Get a move from the AlphaZero model.
    
    Args:
        board: python-chess Board object.
        model: Neural network model.
        game: Game instance.
        num_simulations: Number of MCTS simulations.
        temperature: Temperature for move selection.
        
    Returns:
        The selected move.
    """
    # Convert board to AlphaZero state
    state = board
    
    # Determine player
    player = 1 if board.turn == chess.WHITE else -1
    
    # Get canonical state
    canonical_state = game.change_perspective(state, player)
    
    # Get action distribution from MCTS
    args = {
        'num_simulations': num_simulations,
        'c_puct': 1.0,
    }
    
    action_probs, _ = get_action_distribution(
        game=game,
        state=canonical_state,
        model=model,
        args=args,
        temperature=temperature,
        add_exploration_noise=False
    )
    
    # Get valid moves
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
    if temperature < 0.01:
        # Deterministic selection
        action = np.argmax(masked_probs)
    else:
        # Sample from distribution
        action = np.random.choice(len(masked_probs), p=masked_probs)
    
    # Convert action to chess.Move
    move = game.get_move_from_action(board, action)
    
    # Ensure the move is legal
    if move not in board.legal_moves:
        # Fallback to random legal move
        print("Warning: Selected move is illegal. Choosing random legal move.")
        move = np.random.choice(list(board.legal_moves))
        
    return move


def display_result(board):
    """
    Display the final result of the game.
    
    Args:
        board: Final board state.
    """
    display_board(board)
    
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate! The game is a draw.")
    elif board.is_insufficient_material():
        print("Insufficient material! The game is a draw.")
    elif board.is_fifty_moves():
        print("Fifty-move rule! The game is a draw.")
    elif board.is_repetition():
        print("Threefold repetition! The game is a draw.")
    else:
        print("Game over!")


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize game
    game = ChessGame()
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(args.model_path)
    if args.model_path != 'None':
        model = ResNet.load_checkpoint(args.model_path, game, device)
        print(f"Loaded model from {args.model_path}")
    else:
        model = ResNet(game=game, input_dim=19, device=device)
    
    # Initialize board
    board = chess.Board()
    
    # Set human player
    human_is_white = args.human_player == 'white'
    
    # Game loop
    while not board.is_game_over():
        current_player = 'white' if board.turn == chess.WHITE else 'black'
        print(f"\nCurrent player: {current_player}")
        
        if (human_is_white and board.turn == chess.WHITE) or (not human_is_white and board.turn == chess.BLACK):
            # Human's turn
            move = get_human_move(board)
        else:
            # AlphaZero's turn
            print("AlphaZero is thinking...")
            move = get_ai_move(board, model, game, args.num_simulations, args.temperature)
            print(f"AlphaZero played: {move.uci()}")
        
        # Apply the move
        board.push(move)
        
        # Display the board after AI's move
        if (human_is_white and board.turn == chess.WHITE) or (not human_is_white and board.turn == chess.BLACK):
            display_board(board)
    
    # Display final result
    display_result(board)
    
    # Ask to play again
    play_again = input("\nPlay again? (y/n): ").lower()
    if play_again == 'y':
        # Reset and start again
        main()


if __name__ == '__main__':
    main() 