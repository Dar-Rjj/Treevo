import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Extraction
    # Short-Term Momentum (3-day)
    df['short_raw_return'] = df['close'] / df['close'].shift(3) - 1
    df['price_acceleration'] = (df['close'] / df['close'].shift(1)) / (df['close'].shift(1) / df['close'].shift(2)) - 1
    df['short_momentum'] = (df['short_raw_return'] + df['price_acceleration']) / 2
    
    # Medium-Term Momentum (8-day)
    df['medium_raw_return'] = df['close'] / df['close'].shift(8) - 1
    df['trend_consistency'] = df['close'].rolling(window=7).apply(
        lambda x: (x > x.shift(1)).sum() / 7, raw=False
    )
    df['medium_momentum'] = (df['medium_raw_return'] + df['trend_consistency']) / 2
    
    # Long-Term Momentum (20-day)
    df['long_raw_return'] = df['close'] / df['close'].shift(20) - 1
    df['price_efficiency'] = (df['close'] - df['close'].shift(20)) / df['close'].diff().abs().rolling(window=20).sum()
    df['long_momentum'] = (df['long_raw_return'] + df['price_efficiency']) / 2
    
    # Regime-Aware Volatility Scaling
    # Volatility Regime Detection
    df['short_vol'] = (df['high'] - df['low']).rolling(window=5).std()
    df['medium_vol'] = (df['high'] - df['low']).rolling(window=10).std()
    df['vol_ratio'] = df['short_vol'] / df['medium_vol']
    
    # Adaptive Scaling Factors
    df['high_vol_scaling'] = 1 / (1 + np.exp(-df['vol_ratio']))
    df['low_vol_scaling'] = 1 / (1 + np.exp(df['vol_ratio']))
    df['regime_weight'] = df['high_vol_scaling'] - df['low_vol_scaling']
    
    # Volume-Price Alignment Signals
    # Volume Momentum Analysis
    df['volume_trend'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(
        lambda x: (x > x.shift(1)).sum() / 4, raw=False
    )
    df['volume_stability'] = 1 / (df['volume'].rolling(window=10).std() + 1e-6)
    
    # Price-Volume Divergence
    df['positive_divergence'] = (df['close'] / df['close'].shift(3) - 1) * (df['volume'] / df['volume'].shift(3) - 1)
    df['negative_divergence'] = (np.sign(df['close'] / df['close'].shift(3) - 1) != np.sign(df['volume'] / df['volume'].shift(3) - 1)).astype(float)
    df['divergence_strength'] = np.abs(df['positive_divergence']) * (1 - df['negative_divergence'])
    
    # Nonlinear Combination
    df['momentum_composite'] = np.sign(df['short_momentum'] * df['medium_momentum'] * df['long_momentum']) * \
                              np.power(np.abs(df['short_momentum'] * df['medium_momentum'] * df['long_momentum']), 1/3)
    
    df['volume_confirmation'] = np.tanh(df['volume_trend'] * df['volume_persistence'] * df['volume_stability'])
    
    df['regime_adjustment'] = df['momentum_composite'] * (1 + df['regime_weight'] * df['divergence_strength'])
    
    df['alpha'] = df['regime_adjustment'] * df['volume_confirmation']
    
    return df['alpha']
