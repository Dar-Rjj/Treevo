import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Continuity Gap Factor combining gap analysis with momentum persistence signals
    """
    data = df.copy()
    
    # 1. Gap Analysis Components
    # Overnight gap magnitude and direction
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_direction'] = np.sign(data['overnight_gap'])
    
    # Consecutive same-direction gaps
    data['consecutive_gaps'] = (data['gap_direction'] == data['gap_direction'].shift(1)).astype(int)
    data['consecutive_gap_count'] = data.groupby((data['consecutive_gaps'] == 0).cumsum())['consecutive_gaps'].cumsum() + 1
    
    # Gap size acceleration
    data['gap_acceleration'] = data['overnight_gap'] / (data['overnight_gap'].shift(1).abs() + 1e-8)
    
    # Intraday gap filling behavior
    gap_fill_denom = data['open'] - data['close'].shift(1)
    data['gap_fill_pct'] = np.where(
        gap_fill_denom != 0,
        (data['close'] - data['open']) / gap_fill_denom,
        0
    )
    
    # 2. Momentum Continuity Components
    # Multi-timeframe momentum alignment
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_alignment'] = np.sign(data['momentum_3d']) == np.sign(data['momentum_10d'])
    
    # Rolling momentum correlation (5-day window)
    momentum_corr = []
    for i in range(len(data)):
        if i >= 5:
            window_3d = data['momentum_3d'].iloc[i-4:i+1]
            window_10d = data['momentum_10d'].iloc[i-4:i+1]
            if len(window_3d) > 1 and len(window_10d) > 1:
                corr = np.corrcoef(window_3d, window_10d)[0,1]
                momentum_corr.append(corr if not np.isnan(corr) else 0)
            else:
                momentum_corr.append(0)
        else:
            momentum_corr.append(0)
    data['momentum_correlation'] = momentum_corr
    
    # Volume-weighted momentum strength
    data['volume_momentum'] = data['momentum_3d'] * (data['volume'] / data['volume'].rolling(5).mean())
    
    # 3. Gap-Momentum Interaction
    # Gap size relative to previous day's range
    prev_day_range = data['high'].shift(1) - data['low'].shift(1)
    data['gap_vs_range'] = data['overnight_gap'].abs() / (prev_day_range / data['close'].shift(1) + 1e-8)
    
    # Gap persistence through daily extremes
    gap_up_persist = (data['overnight_gap'] > 0) & (data['low'] > data['close'].shift(1))
    gap_down_persist = (data['overnight_gap'] < 0) & (data['high'] < data['close'].shift(1))
    data['gap_persistence'] = gap_up_persist.astype(int) - gap_down_persist.astype(int)
    
    # Post-gap momentum efficiency
    daily_range = data['high'] - data['low']
    data['post_gap_efficiency'] = np.where(
        data['overnight_gap'] != 0,
        (data['close'] - data['open']).abs() / (daily_range + 1e-8),
        0
    )
    
    # 4. Continuity Breakpoints Detection
    # Momentum fracture - price level where momentum stalls
    data['high_momentum_stall'] = (data['high'] == data['close']) & (data['momentum_3d'] > 0)
    data['low_momentum_stall'] = (data['low'] == data['close']) & (data['momentum_3d'] < 0)
    
    # Volume drop-off at extremes
    volume_ma = data['volume'].rolling(5).mean()
    data['volume_drop_high'] = (data['high'] == data['close']) & (data['volume'] < volume_ma * 0.8)
    data['volume_drop_low'] = (data['low'] == data['close']) & (data['volume'] < volume_ma * 0.8)
    
    # Gap exhaustion signals
    data['gap_exhaustion'] = (
        (data['overnight_gap'].abs() > data['overnight_gap'].rolling(10).mean().abs()) &
        (data['gap_fill_pct'].abs() > 0.8)
    )
    
    # 5. Combined Factor Construction
    # Gap strength classification
    gap_strength = np.tanh(data['gap_vs_range'] * 2) * data['consecutive_gap_count']
    
    # Continuity score
    continuity_score = (
        data['momentum_alignment'].astype(int) * 0.3 +
        np.tanh(data['momentum_correlation'] * 3) * 0.3 +
        np.tanh(data['volume_momentum'] * 10) * 0.2 +
        data['gap_persistence'] * 0.2
    )
    
    # Breakpoint adjustment
    breakpoint_penalty = (
        data['high_momentum_stall'].astype(int) * 0.4 +
        data['low_momentum_stall'].astype(int) * 0.4 +
        data['volume_drop_high'].astype(int) * 0.1 +
        data['volume_drop_low'].astype(int) * 0.1
    )
    
    # Adaptive signal combination
    gap_fill_prob = 1 - data['gap_fill_pct'].abs()
    momentum_continuation = np.tanh(data['post_gap_efficiency'] * 5)
    
    # Final factor
    factor = (
        gap_strength * 0.4 +
        continuity_score * 0.4 -
        breakpoint_penalty * 0.2
    ) * gap_fill_prob * momentum_continuation
    
    # Normalize and handle edge cases
    factor_series = pd.Series(factor, index=data.index)
    factor_series = factor_series.replace([np.inf, -np.inf], np.nan)
    
    return factor_series
