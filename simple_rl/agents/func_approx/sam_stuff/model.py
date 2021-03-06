import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DenseTerminationModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        super(DenseTerminationModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # This gets reshaped to make a state for every action.
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, states, actions, mode="probs"):
        """
        Predicts termination probability for every action... I think.
        So, we should sigmoid so it's between 0 and 1.
        """
        assert mode in ["logits", "probs"]
        assert len(states.shape) == 2, states.shape
        assert len(actions.shape) == 1, actions.shape



        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = x.view(-1, self.state_size, self.action_size)

        # print('Terminal')
        # import ipdb; ipdb.set_trace()
        # print('Terminal')

        x = x.gather(1, actions.unsqueeze(1))

        # print("Here we want to return a single number or list of numbers.")
        # import ipdb; ipdb.set_trace()
        # print("Here we want to return a single number or list of numbers.")

        x = x.squeeze(1)

        # return x
        # x = x[:,action]
        if mode == "logits":
            # Sigmoid! Important for termination...
            return x
        elif mode == "probs":
            return torch.sigmoid(x)
        else:
            raise Exception(f"Shouldn't get here...: {mode}")


class DenseTransitionModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        super(DenseTransitionModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # This gets reshaped to make a state for every action.
        self.fc3 = nn.Linear(fc2_units, action_size * state_size)

    def forward(self, states, actions):
        """
        input: a LIST of states (dim 2)
               a LIST of actions (dim 1)

        Predicts a state for every action... I think.

        action is going to be a list of integers. It wasn't before but now it is.
        """

        assert len(states.shape) == 2, states.shape
        assert len(actions.shape) == 1, actions.shape

        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.state_size, self.action_size)

        ### THIS WORKS TOO. I DON'T UNDERSTAND THE OTHER BUT IT DOES GOOD.
        # a2 = action.squeeze()
        # action_one_hot = F.one_hot(a2, num_classes=self.action_size)
        # action_one_hot = action_one_hot.unsqueeze(-1).float()
        # print(action_one_hot.shape)
        # to_return_onehot = x.bmm(action_one_hot)

        for_gather = actions.unsqueeze(-1).unsqueeze(-1).expand(-1,self.state_size,-1)
        after_gather = x.gather(2, for_gather).squeeze(-1)
        # print(after_gather.shape)
        return after_gather


class DenseRewardModel(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        super(DenseRewardModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # This gets reshaped to make a state for every action.
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, states, actions):
        """
        Predicts a reward for every action... I think.

        It's going to take in a LIST of states and actions. Always.
        """

        assert len(states.shape) == 2, states.shape
        assert len(actions.shape) == 1, actions.shape

        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Expects something like 32,1. It's sort of like where to pluck...

        # print('rewarding!')
        # import ipdb; ipdb.set_trace()
        # print('rewarding')


        x = x.gather(1, actions.unsqueeze(1))

        x = x.squeeze(1)

        return x


class DenseQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """
        Set up the layers of the DQN
        Args:
            state_size (int): number of states in the state variable (can be continuous)
            action_size (int): number of actions in the discrete action domain
            seed (int): random seed
            fc1_units (int): size of the hidden layer
            fc2_units (int): size of the hidden layer
        """
        super(DenseQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        DQN forward pass
        Args:
            state (torch.tensor): convert env.state into a tensor

        Returns:
            logits (torch.tensor): score for each possible action (1, num_actions)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvQNetwork(nn.Module):
    def __init__(self, in_channels=4, n_actions=14, seed=0):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
            seed (int): random seed
        """
        super(ConvQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

        torch.manual_seed(seed)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

    def initialize_with_smaller_network(self, smaller_net, init_q_value=0.):
        """
        Given a DQN over K actions, create a DQN over K + 1 actions. This is needed when we augment the
        MDP with a new action in the form of a learned option.
        Args:
            smaller_net (QNetwork)
            init_q_value (float)
        """
        def copy_layer(bigger, smaller):
            for my_param, source_param in zip(bigger.parameters(), smaller.parameters()):
                my_param.data.copy_(source_param)

        copy_layer(self.conv1, smaller_net.conv1)
        copy_layer(self.conv2, smaller_net.conv2)
        copy_layer(self.conv3, smaller_net.conv3)
        copy_layer(self.fc4, smaller_net.fc4)
        copy_layer(self.bn1, smaller_net.bn1)
        copy_layer(self.bn2, smaller_net.bn2)
        copy_layer(self.bn3, smaller_net.bn3)

        smaller_num_labels = smaller_net.head.out_features
        self.head.weight[:smaller_num_labels, :].data.copy_(smaller_net.head.weight)
        self.head.bias[:smaller_num_labels].data.copy_(smaller_net.head.bias)

        new_action_idx = self.head.out_features - 1
        self.head.bias[new_action_idx].data.fill_(init_q_value)


class ConvInitiationClassifier(nn.Module):
    def __init__(self, device, in_channels=1, seed=0):
        """
        Binary classifier for classifying images I(s) -> {0, 1}
        Args:
            in_channels (int): number of input channels
            seed (int): random seed
        """
        super(ConvInitiationClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, 1)

        torch.manual_seed(seed)
        self.to(device)

    def forward(self, x):
        x = x.unsqueeze(1) # (Batch, Channels, Width, Height)
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = torch.sigmoid(logits)
        return probabilities > 0.5
        self.train()
