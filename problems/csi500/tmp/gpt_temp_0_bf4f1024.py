import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Price Divergence Momentum
    # Compute Price Range Momentum
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['range_momentum'] = df['price_range'].pct_change(periods=3)
    
    # Calculate Volume Efficiency
    df['intraday_move'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_efficiency'] = df['intraday_move'] * df['volume']
    
    # Identify Divergence
    df['divergence'] = df['range_momentum'] - df['volume_efficiency'].pct_change()
    
    # Volume trend persistence
    df['volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else np.nan)
    df['price_action'] = df['close'].pct_change(periods=2)
    
    # Generate Alpha Factor - Volume-Price Divergence Momentum
    alpha_vpdm = df['divergence'] * df['volume_trend'].fillna(0) * np.sign(df['price_action'].fillna(0))
    
    # Volatility-Regime Reversal Strength
    # Identify High Volatility Days
    df['returns'] = df['close'].pct_change()
    df['volatility_20d'] = df['returns'].rolling(window=20).std()
    volatility_quintile = df['volatility_20d'].rolling(window=60, min_periods=20).apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop').iloc[-1] if len(x) >= 20 else np.nan, raw=False)
    high_vol_flag = (volatility_quintile == 4).astype(float)
    
    # Compute Intraday Reversal Signal
    df['reversal_signal'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Apply Regime Filter
    regime_reversal = df['reversal_signal'] * high_vol_flag
    
    # Enhance with Volume Confirmation
    df['volume_20d_avg'] = df['volume'].rolling(window=20).mean()
    volume_weight = df['volume'] / df['volume_20d_avg']
    
    # Volatility-Regime Reversal Strength factor
    alpha_vrrs = regime_reversal * volume_weight
    
    # Amplitude-Volume Consistency
    # Calculate Price Amplitude Trend
    df['amplitude_momentum'] = df['price_range'].pct_change(periods=5)
    
    # Measure Volume Direction Consistency
    df['volume_direction'] = np.sign(df['volume'].diff())
    volume_consistency = df['volume_direction'].rolling(window=5).apply(lambda x: len([i for i in range(1, len(x)) if x[i] == x[i-1]]) / 4 if len(x) == 5 else np.nan)
    
    # Combine Signals
    combined_signal = df['amplitude_momentum'] * volume_consistency
    
    # Apply Recency Weighting
    weights = pd.Series(np.exp(-np.arange(5)/2)[::-1], index=range(5))
    df['weighted_combined'] = combined_signal.rolling(window=5).apply(lambda x: np.average(x, weights=weights) if len(x) == 5 and not np.isnan(x).any() else np.nan)
    
    # Amplitude-Volume Consistency factor
    alpha_avc = df['weighted_combined']
    
    # Combine all three alpha factors with equal weighting
    alpha_combined = (alpha_vpdm.fillna(0) + alpha_vrrs.fillna(0) + alpha_avc.fillna(0)) / 3
    
    return alpha_combined
