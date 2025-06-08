import numpy as np
from .node import Node
import torch
import copy
import concurrent.futures
import math

class MCTS:
    """
    Monte Carlo Tree Search implementation for AlphaZero.
    This version is optimized with BATCHED inference and supports
    MULTI-GPU parallel evaluation.
    """

    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        self.num_virtual_losses = 1
        self.batch_size = self.args.get('mcts_batch_size', 8)
        
        # --- NEW: Multi-GPU Initialization ---
        self.device = self.args.get('device', 'cpu')
        self.thread_pool = None
        self.model_replicas = []
        
        if self.device == 'cuda' and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                # print(f"[MCTS] Found {self.num_gpus} GPUs. Initializing model replicas for parallel inference.")
                # Create a deep copy of the model for each GPU
                # This assumes the model's nn.Module is at self.model.model
                # and the main model is on the CPU or the primary device.
                self.model.model.to('cpu') # Move original to CPU to avoid OOM on device 0
                for i in range(self.num_gpus):
                    replica = copy.deepcopy(self.model)
                    replica.model.to(f'cuda:{i}')
                    self.model_replicas.append(replica)
                
                # Move original model back to the primary device if needed for other tasks
                self.model.model.to('cuda:0')
                
                # Create a thread pool to manage parallel predictions
                self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus)
        else:
            self.num_gpus = 0
        # --- END NEW ---

    def __del__(self):
        """Ensure the thread pool is shut down when the MCTS object is destroyed."""
        if self.thread_pool:
            self.thread_pool.shutdown()

    def _predict_worker(self, model_replica, state_chunk):
        """
        A helper function that runs model prediction in a separate thread.
        Each thread gets its own model replica located on a specific GPU.
        """
        return model_replica.predict(state_chunk)

    def _parallel_predict(self, encoded_states):
        """
        Splits a batch of states, sends them to different GPUs in parallel,
        and reassembles the results.
        """
        # Split the batch of states into chunks for each GPU
        # np.array_split is useful as it handles cases where the batch
        # size is not perfectly divisible by the number of GPUs.
        state_chunks = np.array_split(encoded_states, self.num_gpus)
        
        # Submit prediction jobs to the thread pool
        future_to_chunk_index = {}
        for i, chunk in enumerate(state_chunks):
            if len(chunk) > 0:
                future = self.thread_pool.submit(self._predict_worker, self.model_replicas[i], chunk)
                future_to_chunk_index[future] = i

        # Collect results as they complete
        results = [None] * len(state_chunks)
        for future in concurrent.futures.as_completed(future_to_chunk_index):
            chunk_index = future_to_chunk_index[future]
            try:
                # The result from predict is a tuple: (logits, values, policies)
                results[chunk_index] = future.result()
            except Exception as exc:
                print(f'Chunk {chunk_index} generated an exception: {exc}')
                # Handle exception appropriately, maybe by returning an empty result
                # or re-raising the exception. For now, we'll store None.
        
        # Filter out any failed chunks and reassemble the results in the correct order
        successful_results = [res for res in results if res is not None]
        if not successful_results:
            raise RuntimeError("All parallel prediction workers failed.")

        # Unpack and concatenate the results
        all_logits = np.concatenate([res[0] for res in successful_results], axis=0)
        all_values = np.concatenate([res[1] for res in successful_results], axis=0)
        all_policies = np.concatenate([res[2] for res in successful_results], axis=0)

        return all_logits, all_values, all_policies

    def search(self, state, add_exploration_noise=False):
        """
        Perform MCTS from the given state using batched and potentially parallel evaluations.
        """
        root = Node(self.game, state)

        # First evaluation for the root node is always done on the main model
        encoded_state = self.game.get_encoded_state(state)
        _, _, policy_probs = self.model.predict(encoded_state)
        policy = policy_probs[0]

        if add_exploration_noise:
            self._add_dirichlet_noise(policy, state)

        root.expand(policy)
        root.update(0)

        num_loops = self.args.get('num_simulations', 800) // self.batch_size
        for _ in range(num_loops):
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
                
                # --- MODIFIED: Use parallel prediction if conditions are met ---
                use_parallel = self.thread_pool and len(leaf_nodes_to_evaluate) > self.num_gpus
                
                if use_parallel:
                    _, values, policies = self._parallel_predict(encoded_states)
                else:
                    # Fallback to single-device prediction
                    _, values, policies = self.model.predict(encoded_states)
                # --- END MODIFIED ---

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
    # Don't forget to clean up the MCTS object to shut down the thread pool
    del mcts
    return action_probs, root