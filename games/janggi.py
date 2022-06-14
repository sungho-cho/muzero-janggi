import datetime
import math
import pathlib

import numpy
import torch

import gym
import gym_janggi
from janggi import JanggiGame

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        # Game
        self.observation_shape = (15, 10, 9)  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(9999))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 8  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        # Self-Play
        self.num_workers = 5  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 200  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated (based on paper)
        self.discount = 0.997  # Chronological discount of the reward (based on paper)
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        # Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 8  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 16  # Number of blocks in the ResNet (based on paper)
        self.channels = 64  # Number of channels in the ResNet (based on paper)
        self.reduced_channels_reward = 64  # Number of channels in reward head
        self.reduced_channels_value = 64  # Number of channels in value head
        self.reduced_channels_policy = 64  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        # Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100  # Total number of training steps (ie weights update according to a batch) (based on paper)
        self.batch_size = 1024  # Number of parts of games to train on at each training step (based on paper)
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.1  # Initial learning rate
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        # Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 121  # Number of game moves to keep for every batch element
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value (based on paper)
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        # Values based on paper
        if trained_steps < 500.0 * self.training_steps:
            return 1.0
        elif trained_steps < 750.0 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("gym_janggi/Janggi-v0")

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, info = self.env.step(action)
        if done and "log" in info:
            self.save_log(info["log"])
        return self.get_observation(observation), reward, done

    def get_observation(self, env_observation):
        boards_piece = []
        for i in range(-7, 8):  # 14 planes
            if i == 0:
                continue
            boards_piece.append(numpy.where(env_observation == i, 1.0, 0.0))
        board_to_play = numpy.full(  # 1 plane
            (10, 9), self.env.to_play(), dtype="float32")
        return numpy.array(boards_piece + [board_to_play])

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0 if self.env.to_play() == 1 else 1

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
        return self.get_observation(observation)

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render(mode="ansi")
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)

    def save_log(self, game_log):
        """
        Save game log to logs/ directory.

        Args:
            game_log: Log data to store for future reference.
        """
        if not game_log.move_log:
            return
        print("Saving log...")
        logs_path = pathlib.Path(__file__).resolve().parents[1] / "logs"
        logs_path.mkdir(parents=False, exist_ok=True)
        log_file = logs_path / \
            str(datetime.datetime.today().strftime('%Y-%m-%d.%H:%M:%S'))
        with open(log_file, "wb") as f:
            proto_str = game_log.to_proto().SerializeToString()
            f.write(proto_str)
