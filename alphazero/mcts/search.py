import numpy as np
from .node import Node

class MCTS:
    """
    Monte Carlo Tree Search implementation for AlphaZero.
    This version is optimized with BATCHED inference.
    """

    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        # The number of parallel traversals before a single NN evaluation
        self.num_virtual_losses = 1 # A common hyperparameter for batched MCTS
        self.batch_size = self.args.get('mcts_batch_size', 8) # How many leaves to evaluate at once

    def search(self, state, add_exploration_noise=False):
        """
        Perform MCTS from the given state using batched evaluations.
        """
        # Create root node
        root = Node(self.game, state)

        # First evaluation for the root node
        encoded_state = self.game.get_encoded_state(state)
        _, _, policy_probs = self.model.predict(encoded_state)
        policy = policy_probs[0]

        if add_exploration_noise:
            self._add_dirichlet_noise(policy, state)

        root.expand(policy)
        root.update(0) # Visit the root once

        for _ in range(self.args.get('num_simulations', 800) // self.batch_size):
            leaf_nodes_to_evaluate = []
            
            # 1. SELECTION - Run multiple traversals
            for _ in range(self.batch_size):
                node = root
                
                # Use virtual loss to discourage other traversals from selecting the same path
                # before the true value is backed up.
                self._apply_virtual_loss(node, self.num_virtual_losses)

                # Traverse until a leaf node is found
                while node.is_fully_expanded:
                    action, node = node.select_child(c_puct=self.args.get('c_puct', 1.0))
                    self._apply_virtual_loss(node, self.num_virtual_losses)

                # Check if the game is over
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

                if is_terminal:
                    # For terminal nodes, backup the true value immediately
                    value = self.game.get_opponent_value(value)
                    node.backup(value)
                    self._remove_virtual_loss(node, self.num_virtual_losses)
                else:
                    # This node needs evaluation by the NN
                    leaf_nodes_to_evaluate.append(node)

            # 2. EXPANSION & EVALUATION (in a batch)
            if leaf_nodes_to_evaluate:
                # Prepare batch of states for the neural network
                encoded_states = np.array([self.game.get_encoded_state(n.state) for n in leaf_nodes_to_evaluate])
                
                # Get policies and values from the neural network in one go
                _, values, policies = self.model.predict(encoded_states)

                # 3. BACKUP
                for i, node in enumerate(leaf_nodes_to_evaluate):
                    policy = policies[i]
                    value = values[i][0]
                    
                    # Expand the node
                    node.expand(policy)
                    
                    # Backup the value from the NN
                    node.backup(value)
                    self._remove_virtual_loss(node, self.num_virtual_losses)

        return root

    def _apply_virtual_loss(self, node, num_losses):
        """During selection, temporarily penalize a node to discourage selection by other traversals."""
        # To make the Q-value (which is -node.get_value()) smaller, we need to make node.get_value() larger.
        # We do this by adding to value_sum.
        node.visit_count += num_losses
        node.value_sum += num_losses  # <<< THIS IS THE FIX

    def _remove_virtual_loss(self, node, num_losses):
        """Remove the virtual loss after the true value has been backed up."""
        node.visit_count -= num_losses
        node.value_sum -= num_losses  # <<< THIS IS THE FIX

    def _add_dirichlet_noise(self, policy, state):
        """Adds Dirichlet noise to the policy of the root node for exploration."""
        alpha = self.args.get('dirichlet_alpha', 0.3)
        epsilon = self.args.get('dirichlet_epsilon', 0.25)

        valid_moves = self.game.get_valid_moves(state)
        valid_indices = np.where(valid_moves)[0]
        
        if len(valid_indices) == 0:
            return

        noise = np.random.dirichlet([alpha] * len(valid_indices))
        
        # More efficient way to add noise
        policy[valid_indices] = (1 - epsilon) * policy[valid_indices] + epsilon * noise

# The get_action_distribution function remains the same.
def get_action_distribution(game, state, model, args, temperature=1.0, add_exploration_noise=False):
    mcts = MCTS(model, game, args)
    root = mcts.search(state, add_exploration_noise)
    action_probs = root.get_improved_policy(temperature)
    return action_probs, root