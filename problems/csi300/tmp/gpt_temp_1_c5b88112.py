import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Velocity Component
    # Short-Term Price Velocity
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_acc'] = data['ret_3d'] - data['ret_5d']
    
    # Volume Velocity
    data['vol_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['vol_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_acc'] = data['vol_5d_avg'] - data['vol_10d_avg']
    
    # Volatility Regime Classification
    # Calculate daily returns
    data['daily_ret'] = data['close'].pct_change()
    
    # Short-term volatility (5-day)
    data['vol_5d'] = data['daily_ret'].rolling(window=5).std()
    
    # Medium-term volatility (20-day)
    data['vol_20d'] = data['daily_ret'].rolling(window=20).std()
    
    # Volatility regime
    data['high_vol_regime'] = (data['vol_5d'] > data['vol_20d']).astype(int)
    
    # Divergence Detection System
    # Price-Volume Divergence
    data['divergence_raw'] = data['momentum_acc'] * data['volume_acc']
    
    # Efficiency-Adjusted Divergence
    data['efficiency_ratio'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    data['inefficiency_score'] = 1 - data['efficiency_ratio']
    data['divergence_adj'] = data['divergence_raw'] * data['inefficiency_score']
    
    # Reversal Component
    # Intraday reversal strength
    data['intraday_reversal'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_reversal'] = data['intraday_reversal'].replace([np.inf, -np.inf], np.nan).fillna(0)
    data['intraday_reversal_ewm'] = data['intraday_reversal'].ewm(span=3, adjust=False).mean()
    
    # Oversold/overbought conditions
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    data['range_position'] = (data['close'] - data['low_5d']) / (data['high_5d'] - data['low_5d'])
    data['range_position'] = data['range_position'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    data['range_tanh'] = np.tanh((data['range_position'] - 0.5) * 4)
    
    # Volume-Price Divergence
    data['vol_weighted_ret'] = data['daily_ret'] * data['volume']
    data['vol_weighted_5d'] = data['vol_weighted_ret'].rolling(window=5).sum()
    data['pure_ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['vol_price_div'] = data['vol_weighted_5d'] - data['pure_ret_5d']
    
    # Adaptive Signal Synthesis
    # Core Factor Construction
    data['core_signal'] = data['momentum_acc'] * data['volume_acc'] * data['inefficiency_score']
    
    # Regime-Based Signal Selection
    # High volatility regime: emphasize reversal signals
    high_vol_signal = (
        data['range_tanh'] * -1.0 +  # Reversal emphasis
        data['intraday_reversal_ewm'] * 0.7 +  # Intraday confirmation
        data['core_signal'] * 0.3  # Momentum filter
    )
    
    # Low volatility regime: emphasize momentum signals
    low_vol_signal = (
        data['core_signal'] * 1.2 +  # Momentum emphasis
        data['intraday_reversal_ewm'] * 0.5 +  # Intraday confirmation
        data['ret_5d'] * 0.3  # Medium-term momentum
    )
    
    # Combine signals based on regime
    data['regime_signal'] = (
        data['high_vol_regime'] * high_vol_signal + 
        (1 - data['high_vol_regime']) * low_vol_signal
    )
    
    # Dynamic Smoothing & Enhancement
    # Volatility-dependent window selection
    data['adaptive_window'] = np.where(data['high_vol_regime'] == 1, 5, 10)
    
    # Apply median filtering with adaptive window
    data['final_signal'] = data['regime_signal'].copy()
    
    for i in range(len(data)):
        if i >= 4:  # Ensure we have enough data
            window_size = int(data['adaptive_window'].iloc[i])
            start_idx = max(0, i - window_size + 1)
            window_data = data['regime_signal'].iloc[start_idx:i+1]
            if len(window_data) > 0:
                data['final_signal'].iloc[i] = np.median(window_data)
    
    # Multi-Timeframe Confirmation
    # Validate across different acceleration periods
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    
    # Signal consistency check
    data['signal_consistency'] = (
        (np.sign(data['momentum_3d']) == np.sign(data['momentum_8d'])).astype(int) * 0.2 +
        (np.sign(data['momentum_3d']) == np.sign(data['final_signal'])).astype(int) * 0.8
    )
    
    # Apply consistency enhancement
    data['enhanced_signal'] = data['final_signal'] * data['signal_consistency']
    
    # Final factor output
    factor = data['enhanced_signal']
    
    return factor
