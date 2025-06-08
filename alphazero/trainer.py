import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from .self_play import SelfPlay
from .neural_network.training import AlphaZeroTrainer


class AlphaZero:
    """
    Main AlphaZero trainer class that orchestrates the whole training process.
    """
    
    def __init__(self, game, model, args):
        """
        Initialize AlphaZero trainer.
        
        Args:
            game: Game instance.
            model: Neural network model.
            args: Dictionary of training parameters.
        """
        self.game = game
        self.model = model
        self.args = args
        
        # Set default args if not provided
        if 'num_iterations' not in args:
            self.args['num_iterations'] = 10
        if 'num_selfplay_iterations' not in args:
            self.args['num_selfplay_iterations'] = 100
        if 'num_epochs' not in args:
            self.args['num_epochs'] = 10
        if 'batch_size' not in args:
            self.args['batch_size'] = 128
        if 'device' not in args:
            self.args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        os.makedirs(args.get('checkpoint_dir', 'models'), exist_ok=True)
        os.makedirs(args.get('log_dir', 'logs'), exist_ok=True)
        
        # Initialize neural network trainer
        self.trainer = AlphaZeroTrainer(
            model=self.model,
            lr=args.get('lr', 0.001),
            weight_decay=args.get('weight_decay', 1e-4),
            device=args.get('device', 'cpu')
        )
        
        # Initialize self-play agent
        self.self_play = SelfPlay(
            model=self.model,
            game=self.game,
            args=self.args
        )
        
        # Training history
        self.loss_history = []
        self.selfplay_examples = []
        self.iteration_duration = []
        
    def train(self, start_iteration=0):
        """
        Run the complete AlphaZero training process.
        
        Args:
            start_iteration: The iteration number to start training from (for resuming training).
            
        Returns:
            The trained model.
        """
        print(f"Starting AlphaZero training on {self.args['device']}")
        print(f"Game: {self.game}")
        
        # Adjust total iterations based on start_iteration
        total_iterations = self.args['num_iterations']
        if start_iteration > 0:
            print(f"Resuming training from iteration {start_iteration}")
            total_iterations += start_iteration
        
        for iteration in range(start_iteration, total_iterations):
            print(f"\nIteration {iteration+1}/{total_iterations} (including {start_iteration} warm-up iterations)")
            start_time = time.time()
            
            # 1. Self-play to generate training data
            print("Generating self-play data...")
            iteration_examples = self.self_play.generate_data(
                num_episodes=self.args['num_selfplay_iterations']
            )
            
            # Add examples to the training set
            self.selfplay_examples.extend(iteration_examples)
            
            # Optionally limit the size of the training set to avoid memory issues
            # max_examples = self.args.get('max_examples', 200000)
            max_examples = len(iteration_examples)
            if len(self.selfplay_examples) > max_examples:
                print(f"Limiting training examples to the most recent {max_examples}")
                self.selfplay_examples = self.selfplay_examples[-max_examples:]
                
            # 2. Train neural network with all examples
            print(f"Training neural network on {len(self.selfplay_examples)} examples...")
            iteration_losses = self.trainer.train(
                examples=self.selfplay_examples,
                batch_size=self.args['batch_size'],
                epochs=self.args['num_epochs'],
                log_dir=self.args.get('log_dir', None),
                checkpoint_path=os.path.join(
                    self.args.get('checkpoint_dir', 'models'),
                    f"model_iter{iteration+1}.pt"
                )
            )
            
            # Store training metrics
            self.loss_history.append(np.mean(iteration_losses))
            
            # Record iteration time
            iteration_time = time.time() - start_time
            self.iteration_duration.append(iteration_time)
            print(f"Iteration completed in {iteration_time:.2f} seconds")
            
            # Plot training progress
            self.plot_training_progress()
            
            # Save training state
            self.save_training_state(iteration + 1)
            
        print("\nTraining completed!")
        return self.model
    
    def plot_training_progress(self):
        """
        Plot training loss history.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('AlphaZero Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.args.get('log_dir', 'logs'), 'training_loss.png'))
        plt.close()
        
        # Plot iteration duration
        plt.figure(figsize=(10, 5))
        plt.plot(self.iteration_duration, 'r-')
        plt.xlabel('Iteration')
        plt.ylabel('Duration (s)')
        plt.title('Iteration Duration')
        plt.grid(True)
        plt.savefig(os.path.join(self.args.get('log_dir', 'logs'), 'iteration_duration.png'))
        plt.close()
    
    def save_training_state(self, iteration):
        """
        Save the complete training state.
        
        Args:
            iteration: Current iteration number.
        """
        # Save model
        model_path = os.path.join(self.args.get('checkpoint_dir', 'models'), f"model_iter{iteration}.pt")
        self.model.save_checkpoint(model_path)
        
        # Save optimizer
        optimizer_path = os.path.join(self.args.get('checkpoint_dir', 'models'), f"optimizer_iter{iteration}.pt")
        torch.save(self.trainer.optimizer.state_dict(), optimizer_path)
        
        # Save training metrics
        training_state = {
            'loss_history': self.loss_history,
            'iteration_duration': self.iteration_duration,
            'args': self.args,
            'iteration': iteration,
        }
        
        # Save metrics
        metrics_path = os.path.join(self.args.get('log_dir', 'logs'), 'training_metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(training_state, f)
    
    def load_training_state(self, iteration):
        """
        Load a previously saved training state.
        
        Args:
            iteration: Iteration number to load.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Load model
            model_path = os.path.join(self.args.get('checkpoint_dir', 'models'), f"model_iter{iteration}.pt")
            self.model = type(self.model).load_checkpoint(
                filepath=model_path,
                game=self.game,
                device=self.args.get('device', 'cpu')
            )
            
            # Load optimizer
            optimizer_path = os.path.join(self.args.get('checkpoint_dir', 'models'), f"optimizer_iter{iteration}.pt")
            optimizer_state = torch.load(optimizer_path, map_location=self.args.get('device', 'cpu'))
            self.trainer = AlphaZeroTrainer(
                model=self.model,
                lr=self.args.get('lr', 0.001),
                weight_decay=self.args.get('weight_decay', 1e-4),
                device=self.args.get('device', 'cpu')
            )
            self.trainer.optimizer.load_state_dict(optimizer_state)
            
            # Load metrics
            metrics_path = os.path.join(self.args.get('log_dir', 'logs'), 'training_metrics.pkl')
            with open(metrics_path, 'rb') as f:
                training_state = pickle.load(f)
                
            self.loss_history = training_state['loss_history']
            self.iteration_duration = training_state['iteration_duration']
            
            print(f"Successfully loaded training state from iteration {iteration}")
            return True
            
        except Exception as e:
            print(f"Error loading training state: {e}")
            return False 