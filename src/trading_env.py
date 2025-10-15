import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """A minimal trading environment with discrete actions."""
    
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=30, initial_cash=10_000.0,
                 transaction_cost_pct=0.001, trade_fraction=0.1):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.trade_fraction = trade_fraction

        self.feature_names = ['Close', 'return', 'sma_5', 'sma_20', 'rsi']
        self.n_features = len(self.feature_names) * self.window_size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

        self.current_step = None
        self.cash = None
        self.shares = None
        self.prev_portfolio_value = None

    def _get_obs(self):
        start = self.current_step - self.window_size
        window = self.df.loc[start:self.current_step-1, self.feature_names].copy()
        arr = window.values.astype(np.float32)
        
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        std[std < 1e-8] = 1e-8
        arr = (arr - mean) / std
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = float(self.initial_cash)
        self.shares = 0.0
        self.prev_portfolio_value = float(self.initial_cash)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        price = float(self.df.loc[self.current_step, 'Close'])

        if action == 1:  # Buy
            amount = self.cash * self.trade_fraction
            if amount > 0:
                shares_bought = (amount * (1 - self.transaction_cost_pct)) / price
                self.shares += shares_bought
                self.cash -= amount
        elif action == 2:  # Sell
            shares_to_sell = self.shares * self.trade_fraction
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price * (1 - self.transaction_cost_pct)
                self.shares -= shares_to_sell
                self.cash += proceeds

        self.current_step += 1
        current_price = float(self.df.loc[self.current_step - 1, 'Close'])
        portfolio_value = self.cash + self.shares * current_price

        reward = portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = portfolio_value

        terminated = self.current_step >= len(self.df)
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.n_features, dtype=np.float32)
        info = {
            'portfolio_value': portfolio_value, 
            'cash': self.cash, 
            'shares': self.shares
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Cash: {self.cash:.2f} | "
              f"Shares: {self.shares:.6f} | Portfolio: {self.prev_portfolio_value:.2f}")
