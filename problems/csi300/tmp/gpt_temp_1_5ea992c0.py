import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Gap-Adjusted Momentum
    # Overnight Gap Component
    data['prev_close'] = data['close'].shift(1)
    data['gap_ratio'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Intraday Range Component
    data['intraday_range'] = data['high'] - data['low']
    # Avoid division by zero
    data['intraday_range'] = data['intraday_range'].replace(0, np.nan)
    data['normalized_position'] = (data['close'] - data['low']) / data['intraday_range']
    
    # Combine Gap and Range Signals with Volume Weighting
    data['gap_adjusted_momentum'] = data['gap_ratio'] * data['normalized_position'] * data['volume']
    
    # Calculate Volatility Regime Adjustment
    # Short-term Volatility (5-day)
    data['daily_range'] = data['high'] - data['low']
    data['short_term_vol'] = data['daily_range'].rolling(window=5, min_periods=3).std()
    
    # Medium-term Volatility (20-day)
    data['medium_term_vol'] = data['daily_range'].rolling(window=20, min_periods=10).std()
    
    # Volatility Regime Score
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    data['vol_regime_score'] = np.tanh(data['vol_ratio'])
    
    # Generate Momentum Divergence Signal
    # Short-term Gap Momentum (3-day ROC)
    data['st_gap_momentum'] = data['gap_adjusted_momentum'].pct_change(periods=3) * data['vol_regime_score']
    
    # Medium-term Gap Momentum (10-day ROC)
    data['mt_gap_momentum'] = data['gap_adjusted_momentum'].pct_change(periods=10) * data['vol_regime_score']
    
    # Volume Confirmation
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / data['volume_5d_avg']
    
    # Momentum Divergence
    data['momentum_divergence'] = data['st_gap_momentum'] * data['mt_gap_momentum'] * data['volume_ratio']
    
    # Apply Extreme Gap Filter
    # Identify Significant Gap Events
    data['gap_abs'] = data['gap_ratio'].abs()
    data['gap_percentile'] = data['gap_abs'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Extreme gap flag (top/bottom 15%)
    data['extreme_gap'] = data['gap_percentile'] >= 0.85
    
    # Directional consistency check
    data['gap_direction'] = np.sign(data['gap_ratio'])
    data['intraday_direction'] = np.sign(data['close'] - data['open'])
    data['directional_consistent'] = data['gap_direction'] == data['intraday_direction']
    
    # Enhance signal for extreme gaps
    data['final_factor'] = data['momentum_divergence'].copy()
    
    # Apply enhancement only for extreme gaps with directional consistency
    extreme_mask = data['extreme_gap'] & data['directional_consistent']
    data.loc[extreme_mask, 'final_factor'] = (
        data.loc[extreme_mask, 'momentum_divergence'] * 
        data.loc[extreme_mask, 'gap_abs'] * 2  # Double the effect for extreme gaps
    )
    
    # Return the final factor series
    return data['final_factor']
