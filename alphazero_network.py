import torch
import torch.nn as nn
import torch.nn.functional as F

ABSENT, PRESENT, CORRECT = 0, 1, 2
class AlphaZeroVPNet(nn.Module):
    def __init__(self, word_length: int, max_attempts: int, num_total_valid_words: int, alphabet_size: int = 26):
        super().__init__()

        self.word_length = word_length
        self.max_attempts = max_attempts
        self.num_total_valid_words = num_total_valid_words
        self.alphabet_size = alphabet_size

        self.total_input_features = max_attempts * word_length * alphabet_size

        self.shared_layers = nn.Sequential(
            nn.Linear(self.total_input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_total_valid_words)
        )

        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, state_tensor: torch.Tensor):
        """
        Performs the forward pass through the network.

        Args:
            state_tensor: A tensor of shape (batch_size, total_input_features)
                          representing the game state(s).

        Returns:
            A tuple containing:
            - value_estimate: A tensor of shape (batch_size, 1) with value estimates.
            - policy_logits: A tensor of shape (batch_size, num_total_valid_words) with policy logits.
        """
        shared_output = self.shared_layers(state_tensor)
        policy_logits = self.policy_head(shared_output)
        value_estimate = self.value_head(shared_output)

        return value_estimate, policy_logits

    def evaluate_state(self, state_tensor: torch.Tensor):
        """
        Evaluates a given pre-computed state tensor using the network.

        Args:
            state_tensor: The pre-computed feature tensor representing the state
                          (expected shape: [1, total_input_features] for single eval).

        Returns:
            A tuple containing:
            - value: The scalar value estimate (win probability) for the state.
            - policy_probs: A numpy array of probabilities over all valid actions.
        """
        self.eval()
        if state_tensor.shape[0] != 1:
            state_tensor = state_tensor.unsqueeze(0)

        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            value_estimate, policy_logits = self.forward(state_tensor)

        policy_probs = F.softmax(policy_logits, dim=-1)

        return value_estimate.item(), policy_probs.squeeze(0).cpu().numpy()

