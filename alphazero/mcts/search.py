import numpy as np
from .node import Node
import torch
import copy
import concurrent.futures

class MCTS:
    """
    Monte Carlo Tree Search implementation for AlphaZero.
    This version is optimized with BATCHED inference and supports
    MULTI-GPU parallel evaluation with robust error handling.
    """

    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        self.num_virtual_losses = 1
        self.batch_size = self.args.get('mcts_batch_size', 8)
        
        self.device = self.args.get('device', 'cpu')
        self.thread_pool = None
        self.model_replicas = []
        
        if self.device == 'cuda' and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                # print(f"[MCTS] Found {self.num_gpus} GPUs. Initializing model replicas for parallel inference.")
                self.model.to('cpu') 
                for i in range(self.num_gpus):
                    replica = copy.deepcopy(self.model)
                    device_id = f'cuda:{i}'
                    
                    # Move the replica's parameters to the target device
                    replica.to(device_id)
                    
                    # --- THE FIX ---
                    # Explicitly update the replica's internal device attribute so that its
                    # .predict() method sends data to the correct GPU.
                    # We use hasattr for safety in case the model doesn't have this attribute.
                    if hasattr(replica, 'device'):
                        replica.device = device_id
                    else:
                        print(f"[MCTS Warning] Model replica for {device_id} does not have a 'device' attribute to update.")
                    # --- END FIX ---
                    
                    self.model_replicas.append(replica)
                
                # Move original model back to the primary device
                self.model.to('cuda:0')
                self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus)
        else:
            self.num_gpus = 0

    def __del__(self):
        if self.thread_pool:
            self.thread_pool.shutdown()

    def _predict_worker(self, model_replica, state_chunk):
        return model_replica.predict(state_chunk)

    def _parallel_predict(self, encoded_states):
        """
        Splits a batch, sends to GPUs, and returns results.
        **MODIFICATION**: This function now returns a list of results (or None)
        for each chunk, preserving the structure for error handling.
        """
        state_chunks = np.array_split(encoded_states, self.num_gpus)
        
        future_to_chunk_index = {}
        for i, chunk in enumerate(state_chunks):
            if len(chunk) > 0:
                future = self.thread_pool.submit(self._predict_worker, self.model_replicas[i], chunk)
                future_to_chunk_index[future] = i

        results_by_chunk = [None] * len(state_chunks)
        for future in concurrent.futures.as_completed(future_to_chunk_index):
            chunk_index = future_to_chunk_index[future]
            try:
                results_by_chunk[chunk_index] = future.result()
            except Exception as exc:
                print(f'[MCTS Warning] Prediction on chunk {chunk_index} failed: {exc}')
                # The entry in results_by_chunk remains None
        
        return results_by_chunk

    def search(self, state, add_exploration_noise=False):
        root = Node(self.game, state)

        encoded_state = self.game.get_encoded_state(state)
        _, _, policy_probs = self.model.predict(encoded_state)
        policy = policy_probs[0]

        if add_exploration_noise:
            self._add_dirichlet_noise(policy, state)

        root.expand(policy)
        root.update(0)

        for _ in range(self.args.get('num_simulations', 800) // self.batch_size):
            leaf_nodes_to_evaluate = []
            
            for _ in range(self.batch_size):
                node = root
                self._apply_virtual_loss(node, self.num_virtual_losses)
                while node.is_fully_expanded:
                    action, node = node.select_child(c_puct=self.args.get('c_puct', 1.0))
                    self._apply_virtual_loss(node, self.num_virtual_losses)

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

                if is_terminal:
                    value = self.game.get_opponent_value(value)
                    node.backup(value)
                    self._remove_virtual_loss(node, self.num_virtual_losses)
                else:
                    leaf_nodes_to_evaluate.append(node)

            if leaf_nodes_to_evaluate:
                encoded_states = np.array([self.game.get_encoded_state(n.state) for n in leaf_nodes_to_evaluate])
                
                use_parallel = self.thread_pool and len(leaf_nodes_to_evaluate) > self.num_gpus
                
                if use_parallel:
                    # --- NEW ROBUST BACKUP LOGIC ---
                    # 1. Get results per chunk, which may include Nones for failed chunks
                    prediction_results_by_chunk = self._parallel_predict(encoded_states)

                    # 2. Manually chunk the nodes to align with the prediction chunks
                    # We can't use np.array_split on a list of objects, so we calculate indices.
                    # This logic mimics np.array_split's behavior.
                    total_nodes = len(leaf_nodes_to_evaluate)
                    num_chunks = self.num_gpus
                    base, extra = divmod(total_nodes, num_chunks)
                    chunk_lengths = [base + 1] * extra + [base] * (num_chunks - extra)
                    
                    current_pos = 0
                    node_chunks = []
                    for length in chunk_lengths:
                        node_chunks.append(leaf_nodes_to_evaluate[current_pos : current_pos + length])
                        current_pos += length

                    # 3. Process each chunk
                    for i, chunk_result in enumerate(prediction_results_by_chunk):
                        nodes_in_this_chunk = node_chunks[i]
                        if chunk_result is not None:
                            # This chunk was successful
                            _, values, policies = chunk_result
                            for j, node in enumerate(nodes_in_this_chunk):
                                policy = policies[j]
                                value = values[j][0]
                                node.expand(policy)
                                node.backup(value)
                                self._remove_virtual_loss(node, self.num_virtual_losses)
                        else:
                            # This chunk failed. Just remove the virtual loss for these nodes.
                            for node in nodes_in_this_chunk:
                                self._remove_virtual_loss(node, self.num_virtual_losses)
                    # --- END NEW ROBUST BACKUP LOGIC ---
                else:
                    # Fallback to single-device prediction (original logic)
                    _, values, policies = self.model.predict(encoded_states)
                    for i, node in enumerate(leaf_nodes_to_evaluate):
                        policy = policies[i]
                        value = values[i][0]
                        node.expand(policy)
                        node.backup(value)
                        self._remove_virtual_loss(node, self.num_virtual_losses)

        return root

    def _apply_virtual_loss(self, node, num_losses):
        node.visit_count += num_losses
        node.value_sum += num_losses 

    def _remove_virtual_loss(self, node, num_losses):
        node.visit_count -= num_losses
        node.value_sum -= num_losses

    def _add_dirichlet_noise(self, policy, state):
        alpha = self.args.get('dirichlet_alpha', 0.3)
        epsilon = self.args.get('dirichlet_epsilon', 0.25)
        valid_moves = self.game.get_valid_moves(state)
        valid_indices = np.where(valid_moves)[0]
        
        if len(valid_indices) == 0:
            return

        noise = np.random.dirichlet([alpha] * len(valid_indices))
        policy[valid_indices] = (1 - epsilon) * policy[valid_indices] + epsilon * noise

# The get_action_distribution function remains the same.
def get_action_distribution(game, state, model, args, temperature=1.0, add_exploration_noise=False):
    mcts = MCTS(model, game, args)
    root = mcts.search(state, add_exploration_noise)
    action_probs = root.get_improved_policy(temperature)
    del mcts
    return action_probs, root