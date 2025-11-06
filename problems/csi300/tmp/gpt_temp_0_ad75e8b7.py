import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Momentum Strength
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Intraday Return
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    
    # Momentum Persistence (5-day sum of same-sign intraday returns)
    data['return_sign'] = np.sign(data['intraday_return'])
    data['same_sign_count'] = 0
    
    for i in range(5):
        data['same_sign_count'] += (data['return_sign'] == data['return_sign'].shift(i)).astype(int)
    
    data['momentum_persistence'] = 0
    for i in range(5):
        mask = (data['return_sign'] == data['return_sign'].shift(i))
        data['momentum_persistence'] += np.where(mask, data['intraday_return'].shift(i), 0)
    
    # Assess Market Liquidity Conditions
    # Average Trade Size
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['avg_trade_size'] = data['avg_trade_size'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Stability (10-day coefficient of variation)
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_std_10'] = data['volume'].rolling(window=10, min_periods=5).std()
    data['volume_cv'] = data['volume_std_10'] / data['volume_ma_10']
    
    # Liquidity Score (combining trade size and volume stability)
    data['trade_size_rank'] = data['avg_trade_size'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['volume_stability_rank'] = (1 / (1 + data['volume_cv'])).rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['liquidity_score'] = (data['trade_size_rank'] + data['volume_stability_rank']) / 2
    
    # Identify Momentum Continuation Patterns
    # Strong Momentum Phases (persistence > threshold)
    momentum_threshold = data['momentum_persistence'].rolling(window=20, min_periods=10).quantile(0.7)
    data['strong_momentum'] = (data['momentum_persistence'] > momentum_threshold).astype(int)
    
    # Filter by Liquidity Regime
    liquidity_threshold = data['liquidity_score'].rolling(window=20, min_periods=10).quantile(0.6)
    data['good_liquidity'] = (data['liquidity_score'] > liquidity_threshold).astype(int)
    
    # Momentum Signal weighted by persistence strength
    data['momentum_signal'] = data['momentum_persistence'] * data['strong_momentum'] * data['good_liquidity']
    
    # Incorporate Price-Level Confirmation
    # Distance from Recent High/Low
    data['recent_high_10'] = data['high'].rolling(window=10, min_periods=5).max()
    data['recent_low_10'] = data['low'].rolling(window=10, min_periods=5).min()
    data['distance_from_high'] = (data['close'] - data['recent_high_10']) / data['recent_high_10']
    data['distance_from_low'] = (data['close'] - data['recent_low_10']) / data['recent_low_10']
    
    # Breakout Potential (price level relative to recent range)
    data['recent_range'] = data['recent_high_10'] - data['recent_low_10']
    data['position_in_range'] = (data['close'] - data['recent_low_10']) / data['recent_range']
    data['position_in_range'] = data['position_in_range'].replace([np.inf, -np.inf], 0.5)
    
    # Adjust Momentum Signal based on price level confirmation
    # Stronger signal when near highs (breakout potential) or near lows (reversal potential)
    breakout_strength = np.where(
        data['position_in_range'] > 0.7,  # Near high - breakout potential
        data['position_in_range'],
        np.where(
            data['position_in_range'] < 0.3,  # Near low - reversal potential  
            1 - data['position_in_range'],
            0.5  # Neutral zone
        )
    )
    
    # Final factor combining momentum persistence, liquidity, and price confirmation
    data['factor'] = data['momentum_signal'] * breakout_strength * np.sign(data['intraday_return'])
    
    # Clean up intermediate columns
    result = data['factor'].copy()
    
    return result
