import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import TradingEnv
from src.data_utils import download_data, add_technical_indicators

def main():
    # Parameters
    SYMBOL = 'AAPL'
    START = '2018-01-01'
    END = '2024-12-31'
    WINDOW = 30
    TIMESTEPS = 50_000

    print(f'Training RL agent on {SYMBOL}...')
    
    # Download and prepare data
    df = download_data(SYMBOL, START, END)
    df = add_technical_indicators(df)
    print(f'Data downloaded: {len(df)} rows')

    # Train/test split
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split].reset_index(drop=True)
    
    # Create environment
    env = TradingEnv(df_train, window_size=WINDOW)
    vec_env = DummyVecEnv([lambda: env])

    # Train model
    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/ppo_trader_{SYMBOL}.zip'
    model.save(model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
