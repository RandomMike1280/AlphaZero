import numpy as np
from .node import Node


class MCTS:
    """
    Monte Carlo Tree Search implementation for AlphaZero.
    """
    
    def __init__(self, model, game, args):
        """
        Initialize the MCTS.
        
        Args:
            model: Neural network model.
            game: Game instance.
            args: Dictionary of parameters, including:
                - num_simulations: Number of simulations to run.
                - c_puct: Exploration constant in PUCT formula.
                - dirichlet_alpha: Alpha parameter for Dirichlet noise.
                - dirichlet_epsilon: Epsilon parameter for Dirichlet noise.
        """
        self.model = model
        self.game = game
        self.args = args
        
    def search(self, state, add_exploration_noise=False):
        """
        Perform MCTS from the given state.
        
        Args:
            state: Current game state.
            add_exploration_noise: Whether to add Dirichlet noise to the root's prior probabilities.
            
        Returns:
            Root node of the search tree.
        """
        # Create root node
        root = Node(self.game, state)
        
        # Get policy and value from neural network
        encoded_state = self.game.get_encoded_state(state)
        _, _, policy_probs = self.model.predict(encoded_state)
        policy = policy_probs[0]
        
        # Add Dirichlet noise for exploration (only during self-play)
        if add_exploration_noise:
            self._add_dirichlet_noise(policy, state)
            
        # Expand root node with policy
        root.expand(policy)
        
        # Perform simulations
        for _ in range(self.args.get('num_simulations', 800)):
            node = root
            
            # Selection: Traverse tree to find leaf node
            while node.is_expanded() and node.is_fully_expanded:
                action, node = node.select_child(c_puct=self.args.get('c_puct', 1.0))
            
            # Check if the game is over at this node
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            
            # For terminal nodes, use the game result directly
            if is_terminal:
                # Convert to perspective of parent (previous player)
                value = self.game.get_opponent_value(value)
                node.backup(value)
            else:
                # Expansion and Evaluation
                encoded_state = self.game.get_encoded_state(node.state)
                _, value_pred, policy_probs = self.model.predict(encoded_state)
                policy = policy_probs[0]
                value = value_pred[0][0]  # Extract value from the network output
                
                # Expand the node with policy probs
                node.expand(policy)
                
                # Backup values through the path
                node.backup(value)
            
        return root
    
    def _add_dirichlet_noise(self, policy, state):
        """
        Add Dirichlet noise to the policy at the root node for exploration.
        
        Args:
            policy: Policy probabilities from the neural network.
            state: Current game state.
        """
        alpha = self.args.get('dirichlet_alpha', 0.3)
        epsilon = self.args.get('dirichlet_epsilon', 0.25)
        
        # Only add noise to valid moves
        valid_moves = self.game.get_valid_moves(state)
        valid_move_count = int(np.sum(valid_moves))  # Convert to int for array creation
        
        # Create Dirichlet noise for valid moves only
        noise = np.random.dirichlet([alpha] * valid_move_count)
        
        # Add noise to policy
        noise_idx = 0
        for i in range(len(policy)):
            if valid_moves[i]:
                # Mix policy with noise
                policy[i] = (1 - epsilon) * policy[i] + epsilon * noise[noise_idx]
                noise_idx += 1


def get_action_distribution(game, state, model, args, temperature=1.0, add_exploration_noise=False):
    """
    Get action distribution using MCTS.
    
    Args:
        game: Game instance.
        state: Current game state.
        model: Neural network model.
        args: Dictionary of parameters for MCTS.
        temperature: Temperature for the policy distribution.
        add_exploration_noise: Whether to add Dirichlet noise for exploration.
        
    Returns:
        Tuple of (action distribution, root node).
    """
    # Perform MCTS
    mcts = MCTS(model, game, args)
    root = mcts.search(state, add_exploration_noise)
    
    # Get improved policy based on visit counts
    action_probs = root.get_improved_policy(temperature)
    
    return action_probs, root