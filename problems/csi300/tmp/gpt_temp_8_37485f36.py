import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining volatility regime breakout, intraday momentum reversal,
    gap filling probability, and liquidity-adjusted trend strength.
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Volatility-Regime Range Breakout
    # Calculate rolling volatilities
    data['returns'] = data['close'].pct_change()
    data['vol_5d'] = data['returns'].rolling(window=5).std()
    data['vol_20d'] = data['returns'].rolling(window=20).std()
    
    # Dynamic breakout threshold based on volatility ratio
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    data['breakout_threshold'] = data['vol_ratio'] * data['high'].rolling(window=5).std()
    
    # Range breakout signal
    data['range_breakout'] = (data['high'] - data['low'].rolling(window=5).min()) / data['breakout_threshold']
    
    # Volume confirmation
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_spike'] = data['volume'] / data['volume_20d_avg']
    data['volume_confirmed_breakout'] = data['range_breakout'] * data['volume_spike']
    
    # 2. Intraday Momentum Reversal
    # First hour momentum (assuming first hour is represented by first 25% of day's range)
    data['daily_range'] = data['high'] - data['low']
    data['first_hour_high'] = data['open'] + 0.25 * data['daily_range']
    data['first_hour_low'] = data['open'] - 0.25 * data['daily_range']
    
    # First hour momentum strength
    data['first_hour_momentum'] = (data['close'] - data['first_hour_low']) / (data['first_hour_high'] - data['first_hour_low'])
    
    # Late session reversal (morning vs afternoon volume concentration)
    # Using amount as proxy for intraday volume distribution
    data['morning_volume_ratio'] = data['amount'].rolling(window=5).apply(
        lambda x: x.iloc[0] / x.sum() if x.sum() > 0 else 0, raw=False
    )
    data['late_reversal'] = data['first_hour_momentum'] * (1 - data['morning_volume_ratio'])
    
    # 3. Gap Filling Probability
    # Overnight gap
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Intraday fade (early session volume concentration)
    data['gap_fill_probability'] = -data['overnight_gap'] * data['morning_volume_ratio']
    
    # 4. Liquidity-Adjusted Trend Strength
    # Decay-weighted momentum (exponential weights for multi-period price changes)
    periods = [1, 3, 5, 8]
    weights = [0.4, 0.3, 0.2, 0.1]  # Decaying weights
    
    momentum_components = []
    for i, period in enumerate(periods):
        momentum = (data['close'] - data['close'].shift(period)) / data['close'].shift(period)
        momentum_components.append(momentum * weights[i])
    
    data['decay_momentum'] = sum(momentum_components)
    
    # Volume-price divergence (slopes of price vs volume trends)
    data['price_trend'] = data['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    data['volume_trend'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    data['volume_price_divergence'] = data['price_trend'] * data['volume_trend']
    
    # Liquidity-adjusted trend
    data['liquidity_trend'] = data['decay_momentum'] * (1 + data['volume_price_divergence'])
    
    # Final alpha factor combination
    # Normalize components and combine with equal weights
    components = [
        data['volume_confirmed_breakout'],
        data['late_reversal'], 
        data['gap_fill_probability'],
        data['liquidity_trend']
    ]
    
    # Z-score normalization for each component
    normalized_components = []
    for component in components:
        mean = component.rolling(window=20).mean()
        std = component.rolling(window=20).std()
        normalized = (component - mean) / std
        normalized_components.append(normalized)
    
    # Equal-weighted combination
    alpha_factor = sum(normalized_components) / len(normalized_components)
    
    return alpha_factor
