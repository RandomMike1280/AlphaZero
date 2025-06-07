import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Get the best available device.
    
    Returns:
        'cuda' if GPU is available, 'cpu' otherwise.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize_board(state, game_type='tictactoe'):
    """
    Visualize the game board.
    
    Args:
        state: Game state to visualize.
        game_type: Type of game ('tictactoe' or 'chess').
    """
    if game_type == 'tictactoe':
        plt.figure(figsize=(6, 6))
        plt.imshow(np.zeros((3, 3)), cmap='binary', alpha=0.1, extent=[0, 3, 0, 3])
        
        # Draw grid lines
        for i in range(4):
            plt.axhline(i, color='black', lw=2)
            plt.axvline(i, color='black', lw=2)
        
        # Mark positions
        for i in range(3):
            for j in range(3):
                if state[i, j] == 1:
                    plt.text(j + 0.5, i + 0.5, 'X', fontsize=40, ha='center', va='center')
                elif state[i, j] == -1:
                    plt.text(j + 0.5, i + 0.5, 'O', fontsize=40, ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
    
    elif game_type == 'chess':
        # For chess, we'd use the chess library's SVG rendering
        # This is just a placeholder for now
        print("Chess visualization not implemented yet")


def plot_training_history(metrics_file, save_dir=None):
    """
    Plot training history from a saved metrics file.
    
    Args:
        metrics_file: Path to the metrics file.
        save_dir: Directory to save the plots to. If None, plots are shown instead.
    """
    import pickle
    
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['loss_history'], 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('AlphaZero Training Loss')
    plt.grid(True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'loss_history.png'))
    else:
        plt.show()
    
    # Plot iteration duration
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['iteration_duration'], 'r-')
    plt.xlabel('Iteration')
    plt.ylabel('Duration (s)')
    plt.title('Iteration Duration')
    plt.grid(True)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'iteration_duration.png'))
    else:
        plt.show()


def play_move(model, game, state, player, temperature=0.1):
    """
    Use the model to play a move in the given game state.
    
    Args:
        model: Neural network model.
        game: Game instance.
        state: Current game state.
        player: Current player (1 or -1).
        temperature: Temperature parameter for move selection.
        
    Returns:
        Selected action.
    """
    from .mcts.search import get_action_distribution
    
    # Arguments for MCTS
    args = {
        'num_simulations': 100,  # Use fewer simulations for faster move selection
        'c_puct': 1.0,
    }
    
    # Get canonical state (from current player's perspective)
    canonical_state = game.change_perspective(state, player)
    
    # Get action probabilities from MCTS
    action_probs, _ = get_action_distribution(
        game=game,
        state=canonical_state,
        model=model,
        args=args,
        temperature=temperature,
        add_exploration_noise=False
    )
    
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
    
    return action 