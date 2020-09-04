"""Neural network-based classes for behavioral models."""

import torch
from torch import nn
from device_settings import device
from gumbel_softmax import gumbel_softmax_transform

def sample_outputs(logits, num_actions, gumbel_temp, gumbel_return_hard, detach_gradient=False):
    """Remove gradients from logits and gumbel samples to be produced, if specified."""
    output_unwrapped = logits
    if detach_gradient:
        output_unwrapped = logits.detach()

    # print("output_unwrapped: ", output_unwrapped)

    # Now produce gumbel samples from raw output logits
    # 'batch_gumbelsamples_vectors' is list of length batchsize: each sublist contains one-hot-vectors as tensors

    samples_as_input, sample_vectors = gumbel_softmax_transform(
        output_unwrapped, num_actions, gumbel_temp, gumbel_return_hard
    )
    # print("batch_gumbelsamples_vectors: ", batch_gumbelsamples_vectors) # [[tensor([[0., 1., 0.]])]] for batch size of 1

    # Fully unwrap action tensor
    output_fully_unwrapped = logits[0]
    # print("output_fully_unwrapped.shape: ", output_fully_unwrapped.shape)

    # Example output_fully_unwrapped: tensor([-0.0286, -0.0118,  0.0471], grad_fn=<SelectBackward>)
    #
    # Example batch_gumbelsamples_vectors: [[tensor([[0., 1., 0.]], grad_fn=<AddBackward0>)]]

    # Return tuple (logits tensor, double wrapped gumbel sample tensor)
    return output_fully_unwrapped, sample_vectors


class MLPGeneratorAgent(nn.Module):
    """Feedforward generator agent."""

    def __init__(self, num_actions, history_length, hidden_sizes, num_agents):
        """
        Build feedforward agent.

        Args:
        num_actions: int, number of actions available to agents.
        history_length: int, length of history inputs.
        hidden_sizes: list of ints, number of hidden units in each hidden layer.
        num_agents: number of agents providing inputs to this model, including self.
                    i.e. if only server-client interaction, num_agents=2
        """
        super(MLPGeneratorAgent, self).__init__()

        self._type = "neuralnet"
        self._num_agents = num_agents

        # Defining the layers
        self._num_actions = num_actions
        self._detach_gradient = False
        layers = []
        # An entire history length is the number of actions x num agents x length of interaction.
        self._input_size = nin = (num_actions) * history_length * num_agents
        for size in hidden_sizes:
            layers.append(nn.Linear(nin, size))
            layers.append(nn.ReLU())
            nin = size
        layers.append(nn.Linear(nin, num_actions))
        self.model = nn.Sequential(*layers)
        self.outputWithGradient = True
        self.reset()

        # Record each history of actions and nodal dists
        self._interaction_history = []

    def reset(self):
        """Reset initial state."""
        self.state = None

    def pad(self, x):
        """Pad tensor to fit input size."""
        x = x.to(device)

        batch_size = x.size(0)
        x_size = x.size(1)
        # zero = torch.zeros((batch_size, 1, 1))
        # print(self._input_size, x.size())
        # n_missing_inputs = (self._input_size - x_size - 1) // (self._num_actions + 1)
        # pad = torch.FloatTensor([[0, 0, 0, 1]]).repeat(batch_size, n_missing_inputs).view(batch_size, -1, 1)
        
        zeros = torch.zeros((batch_size, self._input_size - x_size,1)).to(device)
        # print(x.size(), zero.size(), pad.size())
        
        #print("\nzeros shape: ", zeros.shape)
        #print("\nx shape: ", x.shape)

        ## Example of concatenation: 
        #
        # x is [1x3x1]: tensor([[[0.5845],
        #                       [0.5608],
        #                       [0.4611]]])
        #
        # cat([x,zero],dim=1) : tensor([[0.5845, 0.5608, 0.4611, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        #                                0.0000, 0.0000, 0.0000]])
        ##

        padded_x = torch.cat([x, zeros], dim=1).to(device)
        return padded_x
        # return torch.cat([x, zero, pad], dim=1)

    def int_to_ohv_tensor(self, int_action, num_actions):
        """Convert integer to one-hot-vector Tensor."""

        # Create tensor size of action space
        ohv_tensor = torch.zeros(num_actions)

        # Mark the index of integer action, adjusted
        ohv_tensor[int_action-1] = 1

        # Return ohv-tensor
        return ohv_tensor

    def ohv_tensor_to_int(self, ohv_tensor, num_actions):
        """Convert a one-hot-vector tensor to integer action."""

        # Convert tensor to python list and unwrap
        tensor_as_list = ohv_tensor.tolist()[0]

        # Get integer action from index of the 1 in the ohv, adjusted.
        int_action = tensor_as_list.index(1) + 1

        # Return int action
        return int_action

    def act(self, t, inputs, num_actions):
        """
        Format input and return next action.
        Update self interaction history.

        Args:
        inputs: python list of only most recent actions (either mutual or multi-agent).
                i.e. [s1,c1] or [a0_action,a1_action,a2_action]
                actions may be either integers or ohv-tensors
        t: time in the interaction process.
        num_actions: size of the action space.
        """
        #print("\nInside act() with t: ", t, " inputs: ", inputs)

        # Reset model if need be
        if t == 0 or t == 1:
            self.reset()

        # Clear interaction history if starting fresh history
        if t == 0:
            self._interaction_history.clear()

        # For very first action, provide noise input
        if t == 0:
            input_tensor = torch.randn(1,1,1)

        else:
            # Convert input list to tensors if need be
            input_tensor_list = []

            for action in inputs:
                if type(action) != torch.Tensor:
                    action_as_tensor = self.int_to_ohv_tensor(action, num_actions)
                    input_tensor_list.append(action_as_tensor)
                else:
                    input_tensor_list.append(action)

            # Concatenate into single tensor
            input_tensor = torch.cat(input_tensor_list, dim=0)

            # Format input tensor of all actions
            sequence_len = num_actions * self._num_agents 
            input_tensor = input_tensor.view(1, sequence_len, 1)

        #print("\ninput_tensor: ", input_tensor)

        # Call step to statefully provide next action
        (next_logits,next_action) = self.step(input_tensor, num_actions)
        next_action = next_action[0][0] # unwrap
        #print("\nnext_logits: ", next_logits, " next_action: ", next_action)

        # Convert next action to int
        next_action_int = self.ohv_tensor_to_int(next_action, num_actions)

        # Update own interaction history
        self._interaction_history.append((next_action_int,next_logits))

        # Return logits and int action
        return (next_logits,next_action_int)

    # Statefulness: will update state with each action pair from history
    # State stores entire past history so it can be included in forward pass
    def step(self, x, num_actions, gumbel_temp=1.0, gumbel_return_hard=True):
        """Update state and run forward pass."""
       # print("\nInside MLPGeneratorAgent step()!")
       # print("Begin with state: ", self.state)

        # State is not a hidden state like for GRU - this state stores inputs from history as a batch
        if self.state is None:
            self.state = x.to(device)
        else:
            self.state = torch.cat([x, self.state], dim=1).to(device) # most recent action pair at top of input layer
        #print("Updated state: ", self.state)
        #print("Update state shape: ", self.state.shape)

        # Example of self.state (3D tensor) for Ht=[3,1,1,2] where most recent pair is at top of state
        # [[ 
        #    [1]
        #    [0]
        #    [0]
        #    [0]
        #    [1]
        #    [0]
        #
        #    [0]
        #    [0]
        #    [1]
        #    [1]
        #    [0]
        #    [0]
        #   ]]

        return self.forward(self.state, num_actions, gumbel_temp, gumbel_return_hard)

    # Does not modify self.state at all, just a forward pass
    def forward(self, x, num_actions, gumbel_temp, gumbel_return_hard):
        """
        Forward pass on network. 

        Args:
        x: torch tensor with dimensions (batch_size, input_size)
        numActions: int, number of available actions.
        gumbel_temp: float, temperature for gumbel softmax.
        gumbel_return_hard: bool, return

        Returns:
        raw logits and respective gumbel action samples
        """
        padded_x = self.pad(x).view(-1, self._input_size).to(device)  # Pad input if less than right amount.
        # print("\npadded_x shape: ", padded_x.shape)
        # print("padded_x: ", padded_x)

        logits = self.model(padded_x)  # Compute logits with forward pass for single sample
        #print("\nlogits: ", logits)

        # Sample gumbel softmax distribution.
        detach_gradient = not self.outputWithGradient
        unwrapped_outputs, sample_vectors = sample_outputs(
            logits,
            num_actions,
            gumbel_temp,
            gumbel_return_hard,
            detach_gradient=detach_gradient,
        )

        #print("Returning sample_vectors: ", sample_vectors)
        return unwrapped_outputs, sample_vectors

def test_MLPGeneratorAgent():
    """Test methods of MLPGeneratorClass."""

    # Define variables
    num_actions = 3
    history_length = 3 
    hidden_sizes = [5,5]
    num_agents = 3

    mlp = MLPGeneratorAgent(num_actions, history_length, hidden_sizes, num_agents)
    print("\nmlp: ", mlp)

    # Generate integer actions from input
    (logits_t0, action_t0) = mlp.act(t=0, inputs=[], num_actions=num_actions)
    print("\nt = 0: logits_t0: ", logits_t0, " action_t0: ", action_t0)

    # And again
    a1_action_t0 = 1
    a2_action_t0 = 2
    inputs = [action_t0, a1_action_t0, a2_action_t0]
    (logits_t1, action_t1) = mlp.act(t=1, inputs=inputs, num_actions=num_actions)
    print("\nt = 0: logits_t1: ", logits_t1, " action_t1: ", action_t1)

    # Check recorded interaction history
    print("\nmlp history: ", mlp._interaction_history)

#if __name__ == '__main__':
#    test_MLPGeneratorAgent()
