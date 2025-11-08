import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Liquidity Regime Classification
    data['volatility_compression'] = (data['high'] - data['low']) / data['close']
    data['liquidity_efficiency'] = data['amount'] / data['volume']
    
    # Normalize regime indicators using rolling percentiles
    data['compression_norm'] = data['volatility_compression'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    data['efficiency_norm'] = data['liquidity_efficiency'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Regime classification
    conditions = [
        (data['compression_norm'] > 0.5) & (data['efficiency_norm'] > 0),  # Accumulation
        (data['compression_norm'] < -0.5) & (data['efficiency_norm'] > 0),  # Trending
        (data['compression_norm'] < -0.5) & (data['efficiency_norm'] < 0)   # Distribution
    ]
    choices = [1, 2, 3]  # 1: Accumulation, 2: Trending, 3: Distribution
    data['regime'] = np.select(conditions, choices, default=2)  # Default to Trending
    
    # Regime-Specific Momentum Signals
    # Accumulation: Gap persistence
    data['prev_close'] = data['close'].shift(1)
    data['gap_persistence'] = np.where(
        data['open'] > data['prev_close'],
        (data['high'] - data['open']) / (data['open'] - data['prev_close'] + 1e-8),
        0
    )
    
    # Trending: Intraday acceleration
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['acceleration'] = data['intraday_return'] - data['intraday_return'].shift(1)
    
    # Distribution: Breakout quality
    data['high_10'] = data['high'].rolling(window=10, min_periods=5).max()
    data['low_10'] = data['low'].rolling(window=10, min_periods=5).min()
    data['breakout_quality'] = (data['close'] - data['high_10']) / (data['high_10'] - data['low_10'] + 1e-8)
    
    # Volume-Price Divergence Detection
    data['price_trend'] = np.sign(data['close'] - data['close'].shift(5))
    data['volume_trend'] = np.sign(data['volume'] - data['volume'].shift(5))
    
    # Divergence patterns
    data['divergence'] = 0
    data.loc[(data['price_trend'] > 0) & (data['volume_trend'] < 0), 'divergence'] = 1   # Accumulation pattern
    data.loc[(data['price_trend'] < 0) & (data['volume_trend'] > 0), 'divergence'] = -1  # Trending pattern
    data.loc[(data['price_trend'] < 0) & (data['volume_trend'] < 0), 'divergence'] = 2   # Distribution pattern
    
    # Adaptive Alpha Factors
    # Regime-Weighted Momentum
    regime_momentum = np.zeros(len(data))
    
    # Accumulation regime
    acc_mask = data['regime'] == 1
    regime_momentum[acc_mask] = data.loc[acc_mask, 'gap_persistence'] * data.loc[acc_mask, 'volatility_compression']
    
    # Trending regime
    trend_mask = data['regime'] == 2
    regime_momentum[trend_mask] = data.loc[trend_mask, 'acceleration'] * data.loc[trend_mask, 'liquidity_efficiency']
    
    # Distribution regime
    dist_mask = data['regime'] == 3
    regime_momentum[dist_mask] = data.loc[dist_mask, 'breakout_quality'] * (1 - data.loc[dist_mask, 'volatility_compression'])
    
    # Divergence-Adjusted Momentum
    alpha_factor = regime_momentum.copy()
    
    # Apply divergence adjustments
    # Positive divergence in distribution: enhanced signal
    dist_pos_div_mask = (data['regime'] == 3) & (data['divergence'] == 2)
    alpha_factor[dist_pos_div_mask] = alpha_factor[dist_pos_div_mask] * 1.5
    
    # Negative divergence in accumulation: reduced signal
    acc_neg_div_mask = (data['regime'] == 1) & (data['divergence'] == -1)
    alpha_factor[acc_neg_div_mask] = alpha_factor[acc_neg_div_mask] * 0.5
    
    # Trending divergence: moderate penalty
    trend_div_mask = (data['regime'] == 2) & (data['divergence'] == -1)
    alpha_factor[trend_div_mask] = alpha_factor[trend_div_mask] * 0.8
    
    # Create final alpha series
    alpha_series = pd.Series(alpha_factor, index=data.index)
    
    # Remove any potential NaN values from the beginning
    alpha_series = alpha_series.fillna(0)
    
    return alpha_series
