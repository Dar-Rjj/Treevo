import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining microstructure patterns, volume-volatility dislocation,
    price-volume coherence, temporal gaps, and market depth resilience metrics.
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Bid-Ask Imbalance Momentum (approximated via price-volume patterns)
    # High volume with small price movement suggests imbalance persistence
    price_range = (data['high'] - data['low']) / data['close']
    volume_ratio = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    imbalance_momentum = (volume_ratio * (1 - price_range)).rolling(window=5, min_periods=3).mean()
    
    # 2. Volume-Volatility Dislocation
    # Abnormal volume relative to realized volatility
    realized_vol = data['close'].pct_change().rolling(window=10, min_periods=5).std()
    normal_volume = data['volume'].rolling(window=20, min_periods=10).median()
    volume_dislocation = (data['volume'] - normal_volume) / (realized_vol * normal_volume + 1e-8)
    volume_efficiency = (data['close'].pct_change().abs() / (data['volume'] + 1e-8)).rolling(window=10, min_periods=5).mean()
    
    # 3. Price-Volume Fractal Coherence
    # Multi-scale correlation between price movements and volume spikes
    price_changes = data['close'].pct_change()
    volume_changes = data['volume'].pct_change()
    
    # Short-term coherence (intraday)
    short_corr = price_changes.rolling(window=5).corr(volume_changes)
    
    # Medium-term coherence
    medium_corr = price_changes.rolling(window=10).corr(volume_changes)
    
    # Coherence breakdown signal
    coherence_breakdown = (short_corr - medium_corr).rolling(window=5, min_periods=3).mean()
    
    # 4. Temporal Gap Exploitation
    # Overnight vs intraday return patterns
    overnight_returns = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    intraday_returns = (data['close'] - data['open']) / data['open']
    
    # Gap closure momentum
    gap_closure = (overnight_returns - intraday_returns).rolling(window=5, min_periods=3).mean()
    
    # Sequential gap patterns
    gap_persistence = overnight_returns.rolling(window=3).apply(lambda x: np.sign(x).sum() / len(x) if len(x) == 3 else 0)
    
    # 5. Market Depth Resilience Metrics (approximated via price impact)
    # Price impact asymmetry
    up_days = data['close'] > data['open']
    down_days = data['close'] < data['open']
    
    # Price impact on up vs down days
    up_impact = (data['high'][up_days] - data['open'][up_days]) / data['open'][up_days]
    down_impact = (data['open'][down_days] - data['low'][down_days]) / data['open'][down_days]
    
    # Rolling impact asymmetry
    impact_asymmetry = pd.Series(index=data.index, dtype=float)
    for i in range(10, len(data)):
        window_data = data.iloc[i-10:i]
        up_window = window_data['close'] > window_data['open']
        down_window = window_data['close'] < window_data['open']
        
        if up_window.sum() > 2 and down_window.sum() > 2:
            avg_up_impact = ((window_data['high'][up_window] - window_data['open'][up_window]) / window_data['open'][up_window]).mean()
            avg_down_impact = ((window_data['open'][down_window] - window_data['low'][down_window]) / window_data['open'][down_window]).mean()
            impact_asymmetry.iloc[i] = avg_up_impact - avg_down_impact
        else:
            impact_asymmetry.iloc[i] = 0
    
    # Depth recovery (price reversal after large moves)
    large_moves = data['close'].pct_change().abs() > data['close'].pct_change().abs().rolling(window=20).quantile(0.8)
    recovery_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(2, len(data)):
        if large_moves.iloc[i-1]:
            recovery = 1 - abs(data['close'].iloc[i] - data['close'].iloc[i-2]) / abs(data['close'].iloc[i-1] - data['close'].iloc[i-2])
            recovery_signal.iloc[i] = recovery
        else:
            recovery_signal.iloc[i] = 0
    
    # Combine all components with appropriate weights
    factor = (
        0.25 * imbalance_momentum +
        0.20 * volume_dislocation.rolling(window=5).mean() +
        0.15 * (-volume_efficiency) +  # Lower efficiency is better
        0.15 * coherence_breakdown +
        0.10 * gap_closure +
        0.08 * gap_persistence +
        0.07 * impact_asymmetry.rolling(window=5).mean()
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=50, min_periods=20).mean()) / (factor.rolling(window=50, min_periods=20).std() + 1e-8)
    
    return factor
