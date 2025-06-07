import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
from tqdm import trange


class AlphaZeroDataset(Dataset):
    """
    Dataset for AlphaZero training.
    """
    
    def __init__(self, states, policies, values):
        """
        Initialize the dataset.
        
        Args:
            states: List of encoded states.
            policies: List of policy targets.
            values: List of value targets.
        """
        self.states = states
        self.policies = policies
        self.values = values
        
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.states)
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Args:
            idx: Index of the example.
            
        Returns:
            (state, policy, value): Training example.
        """
        return self.states[idx], self.policies[idx], self.values[idx]


class AlphaZeroTrainer:
    """
    Trainer for AlphaZero neural network.
    """
    
    def __init__(self, model, optimizer=None, lr=0.001, weight_decay=1e-4, device='cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The ResNet model.
            optimizer: Optimizer instance (if None, Adam is used).
            lr: Learning rate.
            weight_decay: Weight decay for regularization.
            device: Device to train on ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
            
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        
        # Tensorboard writer
        self.tb_writer = None
        
    def train(self, 
              examples, 
              batch_size=128, 
              epochs=10, 
              log_dir=None, 
              checkpoint_path=None,
              checkpoint_freq=10):
        """
        Train the model on the given examples.
        
        Args:
            examples: List of (encoded_state, policy, value) tuples.
            batch_size: Batch size for training.
            epochs: Number of epochs to train for.
            log_dir: Directory to save tensorboard logs.
            checkpoint_path: Path to save model checkpoints.
            checkpoint_freq: Frequency (in epochs) to save checkpoints.
            
        Returns:
            training_losses: List of losses during training.
        """
        # Set up tensorboard if log_dir is provided
        if log_dir:
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            
        # Prepare dataset
        states, policies, values = zip(*examples)
        states = np.array(states)
        policies = np.array(policies)
        values = np.array(values)
        
        dataset = AlphaZeroDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        training_losses = []
        start_time = time.time()
        
        for epoch in trange(epochs):
            self.model.train()
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []
            
            for batch_idx, (batch_states, batch_policies, batch_values) in enumerate(dataloader):
                # Move data to device
                batch_states = torch.tensor(batch_states, dtype=torch.float32).to(self.device)
                batch_policies = torch.tensor(batch_policies, dtype=torch.float32).to(self.device)
                batch_values = torch.tensor(batch_values, dtype=torch.float32).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                policy_logits, value_pred = self.model(batch_states)
                
                # Calculate losses
                policy_loss = -(batch_policies * torch.log_softmax(policy_logits, dim=1)).sum(1).mean()
                value_loss = self.value_loss_fn(value_pred.flatten(), batch_values)
                
                # Combined loss
                loss = policy_loss + value_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track losses
                epoch_losses.append(loss.item())
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                
            # Calculate average losses for the epoch
            avg_loss = np.mean(epoch_losses)
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)
            training_losses.append(avg_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")
            
            # Log to tensorboard
            if self.tb_writer:
                self.tb_writer.add_scalar('Loss/total', avg_loss, epoch)
                self.tb_writer.add_scalar('Loss/policy', avg_policy_loss, epoch)
                self.tb_writer.add_scalar('Loss/value', avg_value_loss, epoch)
                
            # Save checkpoint
            if checkpoint_path and (epoch + 1) % checkpoint_freq == 0:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                checkpoint_file = f"{os.path.splitext(checkpoint_path)[0]}_ep{epoch+1}.pt"
                self.save_checkpoint(checkpoint_file)
                
        # Save final checkpoint
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            self.save_checkpoint(checkpoint_path)
            
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        # Close tensorboard writer
        if self.tb_writer:
            self.tb_writer.close()
            
        return training_losses
    
    def save_checkpoint(self, filepath):
        """
        Save trainer checkpoint.
        
        Args:
            filepath: Path to save the checkpoint to.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """
        Load trainer from checkpoint.
        
        Args:
            filepath: Path to load the checkpoint from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def adjust_learning_rate(self, lr):
        """
        Adjust the learning rate of the optimizer.
        
        Args:
            lr: New learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr 