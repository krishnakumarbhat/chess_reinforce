import chess
import chess.engine
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # 4672 possible moves in chess
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)

    def reset(self):
        self.board.reset()
        return self._get_obs()

    def step(self, action):
        move = self._action_to_move(action)
        if move in self.board.legal_moves:
            self.board.push(move)
            reward = self._get_reward()
            done = self.board.is_game_over()
            return self._get_obs(), reward, done, {}
        else:
            return self._get_obs(), -1, False, {}

    def _get_obs(self):
        board_state = np.zeros((8, 8, 12), dtype=np.float32)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                board_state[square // 8, square % 8, piece.piece_type - 1] = 1 if piece.color == chess.WHITE else -1
        return board_state

    def _get_reward(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return 0
        else:
            return 0

    def _action_to_move(self, action):
        move_uci = chess.Move.from_uci(chess.SQUARE_NAMES[action // 64] + chess.SQUARE_NAMES[action % 64])
        return move_uci

# Create the environment
env = ChessEnv()
check_env(env)

# Train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("chess_ppo_model")

# Load the model
model = PPO.load("chess_ppo_model")

#testing the model
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
