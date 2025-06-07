import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block for ResNet architecture.
    """
    
    def __init__(self, num_filters):
        """
        Initialize a residual block.
        
        Args:
            num_filters: Number of filters in convolutional layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after passing through the residual block.
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    """
    ResNet architecture for AlphaZero.
    """
    
    def __init__(self, game, input_dim=3, num_resblocks=19, num_filters=256, device='cpu'):
        """
        Initialize the ResNet model.
        
        Args:
            game: Game instance.
            num_resblocks: Number of residual blocks.
            num_filters: Number of filters in convolutional layers.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        self.game = game
        self.device = device
        
        # Input block
        self.conv = nn.Conv2d(input_dim, num_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.resblocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_resblocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=3, padding=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * game.row_count * game.column_count, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self.to(device)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor representing the encoded state.
            
        Returns:
            (policy, value): 
                - policy: Policy logits for each action.
                - value: Value prediction in range [-1, 1].
        """
        x = F.relu(self.bn(self.conv(x)))
        
        for resblock in self.resblocks:
            x = resblock(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, encoded_state):
        """
        Predict policy and value for a single state.
        
        Args:
            encoded_state: Encoded state from game.get_encoded_state().
            
        Returns:
            (policy, value, policy_probs): 
                - policy: Policy logits.
                - value: Value prediction.
                - policy_probs: Softmax of policy logits.
        """
        # Convert to tensor and add batch dimension
        encoded_state_tensor = torch.tensor(encoded_state, dtype=torch.float32).to(self.device)
        if len(encoded_state_tensor.shape) == 3:
            encoded_state_tensor = encoded_state_tensor.unsqueeze(0)
        
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            policy, value = self(encoded_state_tensor)
            policy_probs = F.softmax(policy, dim=1)
        
        return policy.cpu().numpy(), value.cpu().numpy(), policy_probs.cpu().numpy()
    
    def save_checkpoint(self, filepath):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint to.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_resblocks': len(self.resblocks),
            'num_filters': self.conv.out_channels
        }, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath, game, device='cpu'):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to load the checkpoint from.
            game: Game instance.
            device: Device to run the model on ('cpu' or 'cuda').
            
        Returns:
            Loaded ResNet model.
        """
        checkpoint = torch.load(filepath, map_location=device)
        print(checkpoint.keys())
        print(checkpoint['input_dim'])
        print(checkpoint['num_resblocks'])
        print(checkpoint['num_filters'])
        model = cls(
            game=game,
            input_dim=checkpoint['input_dim'],
            num_resblocks=checkpoint['num_resblocks'],
            num_filters=checkpoint['num_filters'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 