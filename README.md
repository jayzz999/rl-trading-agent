# 🤖 Reinforcement Learning Trading Agent

A PPO-based trading agent that learns optimal trading strategies using deep reinforcement learning.

## Features

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: Custom Gymnasium trading environment
- **Technical Indicators**: SMA, RSI, Returns
- **Framework**: Stable-Baselines3
- **Interactive Dashboard**: Streamlit visualization

## Project Structure
```
rl-trading-agent/
├── src/
│   ├── trading_env.py      # Custom Gymnasium environment
│   └── data_utils.py        # Data download and preprocessing
├── models/                  # Trained models
├── train.py                 # Training script
├── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/rl-trading-agent.git
cd rl-trading-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train a Model
```bash
python train.py
```

### Run Dashboard
```bash
streamlit run app.py
```

### Customize Parameters

Edit `train.py` to modify:
- Stock symbol
- Training timesteps
- Window size
- Transaction costs

## Performance

The agent learns to:
- Identify market trends using technical indicators
- Optimize buy/sell/hold decisions
- Maximize portfolio value over time

## Technologies

- Python 3.8+
- Stable-Baselines3
- Gymnasium
- yfinance
- Streamlit
- Plotly

## Future Improvements

- Multi-asset trading
- Portfolio optimization
- Risk management constraints
- Advanced feature engineering
- Transformer-based policy networks

## License

MIT

## Acknowledgments

Built with Stable-Baselines3 and Gymnasium.
