import numpy as np
import time
from tqdm import tqdm
from .mcts.search import MCTS, get_action_distribution


class SelfPlay:
    """
    Self-play module to generate training data.
    """
    
    def __init__(self, model, game, args):
        """
        Initialize self-play.
        
        Args:
            model: Neural network model.
            game: Game instance.
            args: Dictionary of parameters for self-play.
        """
        self.model = model
        self.game = game
        self.args = args
        
    def execute_episode(self):
        """
        Execute a single self-play episode and return the training data.
        
        Returns:
            List of (state, policy, value) tuples.
        """
        training_examples = []
        state = self.game.get_initial_state()
        current_player = 1
        action_history = []
        
        # Keep track of some metrics
        episode_step = 0
        episode_start = time.time()
        
        # Set temperature schedule
        temperature = 1.0
        
        # Play until game is over
        while True:
            episode_step += 1
            
            # Adjust temperature based on move number
            if episode_step < self.args.get('temperature_threshold', 15):
                temperature = 1.0
            else:
                temperature = 0.1
            
            # Convert state to canonical form (current player's perspective)
            canonical_state = self.game.change_perspective(state, current_player)
            
            # Get MCTS action distribution
            action_probs, root = get_action_distribution(
                self.game,
                canonical_state,
                self.model,
                self.args,
                temperature=temperature,
                add_exploration_noise=True
            )
            
            # Store state, policy for training
            encoded_state = self.game.get_encoded_state(canonical_state)
            training_examples.append((encoded_state, action_probs, None))  # Value will be updated later
            
            # Select action based on policy
            if episode_step < self.args.get('temperature_threshold', 15):
                # Sample from distribution during early game for exploration
                action = np.random.choice(len(action_probs), p=action_probs)
            else:
                # Pick most visited node in later stages
                action = np.argmax(action_probs)
            
            # Keep track of action history
            action_history.append(action)
            
            # Execute action
            state = self.game.get_next_state(state, action, current_player)
            
            # Check if game is over
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                # Update all examples with the game outcome
                return_examples = []
                for hist_state, hist_policy, _ in training_examples:
                    # Value is from the perspective of the final player
                    # We need to adjust it based on which player generated the example
                    if current_player == 1:
                        return_examples.append((hist_state, hist_policy, value))
                    else:
                        return_examples.append((hist_state, hist_policy, -value))
                
                # Print some episode stats
                episode_duration = time.time() - episode_start
                print(f"Episode completed in {episode_step} steps ({episode_duration:.2f}s)")
                print(f"Game result: {value} from final player's perspective")
                
                return return_examples
            
            # Switch to the other player
            current_player = self.game.get_opponent(current_player)
    
    def generate_data(self, num_episodes):
        """
        Generate training data from multiple self-play episodes.
        
        Args:
            num_episodes: Number of episodes to play.
            
        Returns:
            List of (state, policy, value) tuples.
        """
        training_data = []
        
        for i in tqdm(range(num_episodes), desc="Self-play episodes"):
            # Play one episode and add examples to training data
            episode_data = self.execute_episode()
            training_data.extend(episode_data)
            
            # Optional: add augmented examples via rotations/reflections
            # Note: this works for Tic Tac Toe but may need modification for other games
            if self.args.get('use_symmetries', True) and hasattr(self.game, 'get_symmetries'):
                symmetry_data = self._get_symmetries(episode_data)
                training_data.extend(symmetry_data)
                
        return training_data
    
    def _get_symmetries(self, examples):
        """
        Generate symmetries (rotations, reflections) of the training examples.
        
        Args:
            examples: List of (encoded_state, policy, value) tuples.
            
        Returns:
            List of symmetric (encoded_state, policy, value) tuples.
        """
        symmetry_examples = []
        
        for encoded_state, policy, value in examples:
            # For Tic Tac Toe, we can use 8 symmetries (4 rotations, each with or without reflection)
            # This is game-specific and should be implemented in the game class
            symmetries = self.game.get_symmetries(encoded_state, policy)
            for sym_state, sym_policy in symmetries:
                symmetry_examples.append((sym_state, sym_policy, value))
                
        return symmetry_examples 