import numpy as np


class Node:
    """
    Node class for Monte Carlo Tree Search.
    """
    
    def __init__(self, game, state, parent=None, action_taken=None, prior=0):
        """
        Initialize a MCTS node.
        
        Args:
            game: Game instance.
            state: Current game state.
            parent: Parent node.
            action_taken: Action that led to this node.
            prior: Prior probability from the neural network.
        """
        self.game = game
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = {}  # Maps actions to child nodes
        
        self.visit_count = 0
        self.value_sum = 0
        
        self.expandable = True  # Node can be expanded
        self.is_fully_expanded = False  # All possible child nodes have been created
        
    def is_expanded(self):
        """
        Check if the node has been expanded.
        
        Returns:
            True if the node has been expanded, False otherwise.
        """
        return len(self.children) > 0
    
    def get_value(self):
        """
        Get the average value of the node.
        
        Returns:
            The average value if the node has been visited, 0 otherwise.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.0):
        """
        Select a child node using the PUCT formula.
        
        Args:
            c_puct: Exploration constant in PUCT formula.
            
        Returns:
            Tuple of (action, child node) with highest PUCT value.
        """
        # Calculate UCB scores for all children
        ucb_scores = {}
        for action, child in self.children.items():
            # PUCT formula with transformation like in example implementation
            if child.visit_count == 0:
                q_value = 0
            else:
                # Transform the value similar to the example implementation
                q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
                
            exploration = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            ucb_scores[action] = q_value + exploration
            
        # Select action with highest UCB score
        best_action = max(ucb_scores, key=ucb_scores.get)
        return best_action, self.children[best_action]
    
    def expand(self, policy):
        """
        Expand the node with policy probabilities from the neural network.
        
        Args:
            policy: Policy probabilities for all actions from the neural network.
            
        Returns:
            List of created child nodes.
        """
        if not self.expandable:
            return []
            
        valid_moves = self.game.get_valid_moves(self.state)
        valid_policy = policy * valid_moves  # Mask invalid moves
        
        # Normalize policy to sum to 1
        policy_sum = np.sum(valid_policy)
        if policy_sum > 1e-10:  # Small epsilon to avoid floating point errors
            valid_policy = valid_policy / policy_sum
        else:
            # Fallback: use uniform policy if all valid moves have 0 probability
            valid_moves_sum = np.sum(valid_moves)
            if valid_moves_sum > 0:
                valid_policy = valid_moves / valid_moves_sum
            else:
                # If no valid moves, return zeros (this should be handled by the game logic)
                valid_policy = np.zeros_like(valid_moves)
        
        new_children = []
        for action in range(len(valid_moves)):
            if valid_moves[action]:
                if action not in self.children:
                    # Create a new state (copy current state to avoid modifying it)
                    next_state = self.game.get_next_state(self.state.copy(), action, 1)
                    next_state = self.game.change_perspective(next_state, -1)
                    
                    child = Node(
                        game=self.game,
                        state=next_state,
                        parent=self,
                        action_taken=action,
                        prior=valid_policy[action]
                    )
                    
                    self.children[action] = child
                    new_children.append(child)
        
        # Mark as expanded as soon as we have any children (like in example)
        if len(self.children) > 0:
            self.is_fully_expanded = True
        
        # If we've created all possible children, mark as not expandable anymore
        if len(self.children) == np.sum(valid_moves):
            self.expandable = False
        
        return new_children
    
    def update(self, value):
        """
        Update statistics of the node.
        
        Args:
            value: Value to add to the node's value sum.
        """
        self.visit_count += 1
        self.value_sum += value
        
    def backup(self, value):
        """
        Backup values up the tree.
        
        Args:
            value: Value to propagate.
        """
        # Value should be from the perspective of the parent (opponent)
        current = self
        while current:
            current.update(value)
            value = -value  # Flip value for parent (opponent's perspective)
            current = current.parent
            
    def get_visit_counts(self):
        """
        Get visit count distribution for all actions.
        
        Returns:
            Numpy array of visit counts for all possible actions.
        """
        counts = np.zeros(self.game.action_size)
        for action, child in self.children.items():
            counts[action] = child.visit_count
        return counts
    
    def get_improved_policy(self, temperature=1.0):
        """
        Get improved policy based on visit counts.
        
        Args:
            temperature: Temperature for visit count distribution.
                         1.0 = Normal, <1.0 = More deterministic, >1.0 = More exploratory.
                         
        Returns:
            Improved policy based on MCTS visit counts.
        """
        visit_counts = self.get_visit_counts()
        
        if temperature == 0:  # Deterministic selection
            action = np.argmax(visit_counts)
            improved_policy = np.zeros_like(visit_counts)
            improved_policy[action] = 1.0
            return improved_policy
        
        # Apply temperature
        counts = visit_counts ** (1.0 / temperature)
        total = np.sum(counts)
        
        # Handle case where no visits have occurred
        if total == 0:
            valid_moves = self.game.get_valid_moves(self.state)
            valid_moves_sum = np.sum(valid_moves)
            if valid_moves_sum > 1e-10:  # Small epsilon to avoid floating point errors
                return valid_moves / valid_moves_sum
            # If no valid moves, return uniform distribution (this should be handled by the game logic)
            return np.ones_like(valid_moves) / len(valid_moves) if len(valid_moves) > 0 else valid_moves
            
        return counts / total