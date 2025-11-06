import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize factor series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price features
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_ma'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Regime identification features
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['bull_streak'] = df['higher_high'].rolling(window=5, min_periods=3).sum()
    df['bear_streak'] = df['lower_low'].rolling(window=5, min_periods=3).sum()
    
    # Momentum and strength indicators
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['volatility'] = df['returns'].rolling(window=20, min_periods=10).std()
    
    # Volume-price divergence
    df['price_volume_corr'] = df['returns'].rolling(window=10, min_periods=5).corr(df['volume_ratio'])
    
    # Intraday reversal signals
    df['intraday_reversal'] = ((df['close'] < df['open']) & (df['high'] > df['high'].shift(1))).astype(int) - \
                             ((df['close'] > df['open']) & (df['low'] < df['low'].shift(1))).astype(int)
    
    # Support and resistance tests
    df['resistance_test'] = ((df['high'] > df['high'].rolling(window=10, min_periods=5).max().shift(1)) & 
                            (df['close'] < df['open'])).astype(int)
    df['support_test'] = ((df['low'] < df['low'].rolling(window=10, min_periods=5).min().shift(1)) & 
                         (df['close'] > df['open'])).astype(int)
    
    # Regime transition probability components
    for i in range(len(df)):
        if i < 20:  # Need sufficient history
            factor.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev_data = df.iloc[:i+1]  # Only historical data
        
        # Bullish regime strength
        bull_strength = (
            prev_data['bull_streak'].iloc[-5:].mean() * 0.3 +
            (prev_data['close_position'].iloc[-5:].mean() > 0.6) * 0.2 +
            (prev_data['momentum_5'].iloc[-1] > 0) * 0.2 +
            (prev_data['price_volume_corr'].iloc[-1] > 0) * 0.3
        )
        
        # Bearish regime strength
        bear_strength = (
            prev_data['bear_streak'].iloc[-5:].mean() * 0.3 +
            (prev_data['close_position'].iloc[-5:].mean() < 0.4) * 0.2 +
            (prev_data['momentum_5'].iloc[-1] < 0) * 0.2 +
            (prev_data['price_volume_corr'].iloc[-1] < 0) * 0.3
        )
        
        # Transition warning signals
        volume_divergence = (
            (current['volume_ratio'] > 1.2) * (current['returns'] < 0) * 0.4 -
            (current['volume_ratio'] > 1.2) * (current['returns'] > 0) * 0.4
        )
        
        momentum_deceleration = (
            (abs(current['momentum_5']) < abs(current['momentum_10'])) * 
            np.sign(current['momentum_5']) * 0.3
        )
        
        reversal_signals = (
            current['intraday_reversal'] * 0.4 +
            current['resistance_test'] * -0.3 +
            current['support_test'] * 0.3
        )
        
        # Alpha factor calculation
        regime_diff = bull_strength - bear_strength
        transition_prob = (
            volume_divergence +
            momentum_deceleration +
            reversal_signals +
            (abs(regime_diff) < 0.3) * np.sign(regime_diff) * -0.4  # Weak regime transition
        )
        
        # Risk adjustment
        volatility_adj = 1.0 / (1.0 + current['volatility'])
        
        factor.iloc[i] = transition_prob * volatility_adj
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
