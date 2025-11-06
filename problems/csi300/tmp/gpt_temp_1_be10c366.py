import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum-Adjusted Volume Divergence
    # Raw Price Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Volume Trend
    data['volume_avg_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_avg_10d'] = data['volume'].rolling(window=10).mean()
    
    # Divergence Patterns
    data['volume_trend_5d'] = data['volume'] / data['volume_avg_5d'] - 1
    data['volume_trend_10d'] = data['volume'] / data['volume_avg_10d'] - 1
    
    # Momentum-Volume Divergence Signal
    data['momentum_volume_div'] = np.where(
        (data['momentum_5d'] > 0) & (data['volume_trend_5d'] < 0),
        data['momentum_5d'] * abs(data['volume_trend_5d']),
        np.where(
            (data['momentum_5d'] < 0) & (data['volume_trend_5d'] > 0),
            data['momentum_5d'] * abs(data['volume_trend_5d']),
            0
        )
    )
    
    # Intraday Volatility Persistence
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # Volatility Clustering
    data['volatility_3d_autocorr'] = data['daily_range'].rolling(window=3).apply(
        lambda x: x.autocorr() if len(x) == 3 and not x.isna().any() else 0
    )
    data['volatility_momentum_5d'] = data['daily_range'] / data['daily_range'].shift(5) - 1
    
    # Volume-volatility relationship
    data['volume_vol_ratio'] = data['volume'] / data['daily_range'].replace(0, np.nan)
    data['volume_vol_ratio'] = data['volume_vol_ratio'].fillna(0)
    
    # Volatility persistence signal
    data['vol_persistence_signal'] = data['volatility_3d_autocorr'] * data['volatility_momentum_5d']
    
    # Price-Volume Efficiency Ratio
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['price_efficiency'] = data['price_efficiency'].fillna(0)
    
    # Volume confirmation
    data['efficiency_volume_ratio'] = data['price_efficiency'] * data['volume']
    data['efficiency_trend'] = data['price_efficiency'].rolling(window=5).mean()
    
    # Efficiency signal
    data['efficiency_signal'] = data['efficiency_trend'] * np.sign(data['efficiency_volume_ratio'])
    
    # Amplitude-Frequency Oscillation
    data['daily_amplitude'] = (data['high'] - data['low']) / data['close']
    
    # Frequency of direction changes
    data['price_change'] = data['close'].diff()
    data['direction_change'] = (np.sign(data['price_change']) != np.sign(data['price_change'].shift(1))).astype(int)
    data['direction_freq_5d'] = data['direction_change'].rolling(window=5).mean()
    
    # Oscillation patterns
    data['amplitude_freq_ratio'] = data['daily_amplitude'] / data['direction_freq_5d'].replace(0, np.nan)
    data['amplitude_freq_ratio'] = data['amplitude_freq_ratio'].fillna(0)
    
    # Volume during high amplitude periods
    data['high_amplitude_volume'] = np.where(
        data['daily_amplitude'] > data['daily_amplitude'].rolling(window=10).mean(),
        data['volume'],
        0
    )
    
    # Oscillation signal
    data['oscillation_signal'] = data['amplitude_freq_ratio'] * data['high_amplitude_volume']
    
    # Liquidity-Adjusted Breakout
    # Price breakout strength
    data['recent_high_20d'] = data['high'].rolling(window=20).max()
    data['recent_low_20d'] = data['low'].rolling(window=20).min()
    
    data['breakout_high'] = (data['close'] - data['recent_high_20d']) / data['recent_high_20d']
    data['breakout_low'] = (data['close'] - data['recent_low_20d']) / data['recent_low_20d']
    
    # Liquidity conditions
    data['volume_ratio_10d'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['amount_per_trade'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['amount_per_trade'] = data['amount_per_trade'].fillna(0)
    
    # Breakout-liquidity relationship
    data['liquidity_breakout_high'] = data['breakout_high'] * data['volume_ratio_10d']
    data['liquidity_breakout_low'] = data['breakout_low'] * data['volume_ratio_10d']
    
    # Final breakout signal
    data['breakout_signal'] = np.where(
        data['breakout_high'] > 0,
        data['liquidity_breakout_high'],
        np.where(
            data['breakout_low'] < 0,
            data['liquidity_breakout_low'],
            0
        )
    )
    
    # Combine all signals with weights
    data['combined_alpha'] = (
        0.25 * data['momentum_volume_div'] +
        0.20 * data['vol_persistence_signal'] +
        0.20 * data['efficiency_signal'] +
        0.15 * data['oscillation_signal'] +
        0.20 * data['breakout_signal']
    )
    
    # Normalize the final alpha signal
    alpha_series = data['combined_alpha']
    
    return alpha_series
