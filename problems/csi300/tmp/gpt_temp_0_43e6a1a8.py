import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for momentum and other calculations
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Multiple Timeframe Momentum
    df['short_term_momentum'] = (df['close'] - df['close'].shift(4)) / df['close'].shift(4)
    df['medium_term_momentum'] = (df['close'] - df['close'].shift(9)) / df['close'].shift(9)
    df['long_term_momentum'] = (df['close'] - df['close'].shift(19)) / df['close'].shift(19)
    
    # Momentum Divergence Patterns
    df['momentum_gradient'] = (df['short_term_momentum'] - df['medium_term_momentum']) - (df['medium_term_momentum'] - df['long_term_momentum'])
    df['divergence_strength'] = abs(df['short_term_momentum'] - df['medium_term_momentum']) + abs(df['medium_term_momentum'] - df['long_term_momentum'])
    
    # Volume-Price Efficiency
    # Volume-weighted returns (10-day)
    volume_weighted_returns = []
    for i in range(len(df)):
        if i < 10:
            volume_weighted_returns.append(np.nan)
            continue
        window_returns = df['returns'].iloc[i-9:i+1].values
        window_volume = df['volume'].iloc[i-9:i+1].values
        weighted_sum = np.nansum(window_volume * window_returns)
        volume_weighted_returns.append(weighted_sum)
    df['volume_weighted_returns'] = volume_weighted_returns
    
    # Efficiency ratio (10-day)
    efficiency_ratio = []
    for i in range(len(df)):
        if i < 10:
            efficiency_ratio.append(np.nan)
            continue
        window_returns = df['returns'].iloc[i-9:i+1].values
        volume_weighted = df['volume_weighted_returns'].iloc[i]
        simple_sum = np.nansum(window_returns)
        if simple_sum == 0:
            efficiency_ratio.append(0)
        else:
            efficiency_ratio.append(volume_weighted / simple_sum)
    df['efficiency_ratio'] = efficiency_ratio
    
    # Volume-return correlation (15-day)
    volume_return_corr = []
    for i in range(len(df)):
        if i < 15:
            volume_return_corr.append(np.nan)
            continue
        window_volume = df['volume'].iloc[i-14:i+1].values
        window_returns = df['returns'].iloc[i-14:i+1].values
        valid_mask = ~(np.isnan(window_volume) | np.isnan(window_returns))
        if np.sum(valid_mask) < 5:
            volume_return_corr.append(0)
        else:
            corr_val = np.corrcoef(window_volume[valid_mask], window_returns[valid_mask])[0, 1]
            volume_return_corr.append(corr_val if not np.isnan(corr_val) else 0)
    df['volume_return_correlation'] = volume_return_corr
    
    # Combine Signals
    df['momentum_efficiency_score'] = df['momentum_gradient'] * df['efficiency_ratio']
    df['volume_confirmation'] = df['divergence_strength'] * df['volume_return_correlation']
    
    # Final Composite Score
    df['composite_score'] = df['momentum_efficiency_score'] + df['volume_confirmation']
    
    # Volatility Scaling with Average True Range (10-day)
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 10-day ATR
    df['atr_10day'] = df['true_range'].rolling(window=10, min_periods=5).mean()
    
    # Final alpha factor with volatility scaling
    df['alpha_factor'] = df['composite_score'] * df['atr_10day']
    
    # Clean up intermediate columns
    result = df['alpha_factor'].copy()
    
    return result
