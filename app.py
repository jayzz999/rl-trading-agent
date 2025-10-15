import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from stable_baselines3 import PPO
from src.trading_env import TradingEnv
from src.data_utils import download_data, add_technical_indicators
import os

st.set_page_config(page_title="RL Trading Agent", page_icon="📈", layout="wide")

st.title("🤖 Reinforcement Learning Trading Agent")
st.write("**PPO-based trading agent trained with Stable-Baselines3**")

# Sidebar
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.sidebar.button("🚀 Run Backtest", type="primary"):
    with st.spinner(f"Downloading {symbol} data..."):
        try:
            df = download_data(symbol, str(start_date), str(end_date))
            df = add_technical_indicators(df)
            st.success(f"✅ Downloaded {len(df)} days of data")
            
            # Split data
            split = int(len(df) * 0.8)
            df_test = df.iloc[split:].reset_index(drop=True)
            
            # Check if model exists
            model_path = f'models/ppo_trader_{symbol}.zip'
            if not os.path.exists(model_path):
                st.warning(f"No trained model found for {symbol}. Using demo model (AAPL).")
                model_path = 'models/ppo_trader_AAPL.zip'
            
            if os.path.exists(model_path):
                model = PPO.load(model_path)
                
                # Run backtest
                test_env = TradingEnv(df_test, window_size=30)
                obs, _ = test_env.reset()
                
                portfolio_values = []
                actions_taken = []
                
                with st.spinner("Running agent..."):
                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = test_env.step(int(action))
                        portfolio_values.append(info['portfolio_value'])
                        actions_taken.append(int(action))
                        if terminated or truncated:
                            break
                
                # Calculate metrics
                initial_value = float(portfolio_values[0])
                final_value = float(portfolio_values[-1])
                total_return = ((final_value - initial_value) / initial_value) * 100
                
                # Buy and hold comparison
                first_price = float(df_test['Close'].iloc[0])
                last_price = float(df_test['Close'].iloc[-1])
                buy_hold_return = ((last_price - first_price) / first_price) * 100
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Agent Return", f"{total_return:.2f}%")
                col2.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                col3.metric("Outperformance", f"{total_return - buy_hold_return:.2f}%")
                
                # Plot portfolio value
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=portfolio_values, 
                    mode='lines',
                    name='RL Agent Portfolio',
                    line=dict(color='green', width=2)
                ))
                
                # Add buy & hold baseline
                buy_hold_values = []
                for i in range(len(portfolio_values)):
                    price_change = (float(df_test['Close'].iloc[i]) - first_price) / first_price
                    buy_hold_values.append(initial_value * (1 + price_change))
                
                fig.add_trace(go.Scatter(
                    y=buy_hold_values,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Portfolio Value Over Time - {symbol}",
                    xaxis_title="Trading Days",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Action distribution
                st.subheader("📊 Trading Actions")
                action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
                action_series = pd.Series(actions_taken)
                action_counts = action_series.value_counts()
                
                action_data = []
                for action_num in [0, 1, 2]:
                    count = int(action_counts.get(action_num, 0))
                    action_data.append({'Action': action_names[action_num], 'Count': count})
                
                action_df = pd.DataFrame(action_data)
                st.bar_chart(action_df.set_index('Action'))
                
            else:
                st.error("No model available. Please train a model first using train.py")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Info section
with st.expander("ℹ️ About This Agent"):
    st.markdown("""
    **Technologies:**
    - 🤖 PPO (Proximal Policy Optimization) from Stable-Baselines3
    - 🏋️ Custom Gymnasium trading environment
    - 📊 Technical indicators: SMA, RSI, Returns
    - 💹 Yahoo Finance data
    
    **Actions:**
    - Hold: Do nothing
    - Buy: Purchase 10% of available cash
    - Sell: Sell 10% of holdings
    
    **Features:**
    - 30-day lookback window
    - Transaction costs: 0.1%
    - Reward: Change in portfolio value
    """)
