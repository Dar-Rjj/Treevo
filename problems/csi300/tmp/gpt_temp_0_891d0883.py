import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Efficiency-Adjusted Momentum Breakout factor
    Combines momentum quality, volume efficiency, and breakout context
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Momentum Quality Assessment
    # Multi-timeframe Momentum Alignment
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_8d'] = df['close'] / df['close'].shift(8) - 1
    
    # Momentum Consistency (sign alignment × magnitude ratio)
    momentum_alignment = np.sign(df['momentum_3d']) * np.sign(df['momentum_8d'])
    momentum_ratio = np.abs(df['momentum_3d']) / (np.abs(df['momentum_8d']) + 1e-8)
    df['momentum_consistency'] = momentum_alignment * np.minimum(momentum_ratio, 2.0)
    
    # Momentum Sustainability Scoring
    df['momentum_persistence'] = df['momentum_3d'].rolling(window=5).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 5 else np.nan
    )
    
    # Momentum acceleration/deceleration pattern
    df['momentum_accel'] = df['momentum_3d'] - df['momentum_3d'].shift(3)
    
    # Volume-Efficiency Confirmation
    # Intraday Range Efficiency Analysis
    df['daily_range'] = df['high'] - df['low']
    df['range_efficiency'] = (df['close'] - df['open']) / (df['daily_range'] + 1e-8)
    
    # 5-day Efficiency Persistence (directional consistency)
    df['efficiency_persistence'] = df['range_efficiency'].rolling(window=5).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 5 else np.nan
    )
    
    # Efficiency Regime Shifts (current vs 10-day pattern)
    df['efficiency_10d_avg'] = df['range_efficiency'].rolling(window=10).mean()
    df['efficiency_regime_shift'] = df['range_efficiency'] - df['efficiency_10d_avg']
    
    # Volume Breakout Quality
    df['volume_20d_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_breakout_ratio'] = df['volume'] / (df['volume_20d_avg'] + 1e-8) - 1
    
    # Volume-Momentum Alignment
    volume_momentum_alignment = np.sign(df['volume_breakout_ratio']) * np.sign(df['momentum_3d'])
    df['volume_support_score'] = volume_momentum_alignment * np.minimum(
        np.abs(df['volume_breakout_ratio']), 2.0
    )
    
    # Volume Persistence
    df['volume_persistence'] = df['volume_breakout_ratio'].rolling(window=3).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 3 else np.nan
    )
    
    # Volume acceleration consistency
    df['volume_accel'] = df['volume_breakout_ratio'] - df['volume_breakout_ratio'].shift(3)
    
    # Breakout Context Integration
    # Range Breakout Detection
    df['high_15d_max'] = df['high'].rolling(window=15, min_periods=1).apply(
        lambda x: x[:-1].max() if len(x) == 15 else np.nan
    )
    df['low_15d_min'] = df['low'].rolling(window=15, min_periods=1).apply(
        lambda x: x[:-1].min() if len(x) == 15 else np.nan
    )
    
    upper_breakout = df['high'] > df['high_15d_max']
    lower_breakout = df['low'] < df['low_15d_min']
    
    df['range_breakout'] = 0
    df.loc[upper_breakout, 'range_breakout'] = 1
    df.loc[lower_breakout, 'range_breakout'] = -1
    
    # Breakout Magnitude relative to range
    df['breakout_magnitude'] = 0
    df.loc[upper_breakout, 'breakout_magnitude'] = (df['high'] - df['high_15d_max']) / (df['high_15d_max'] + 1e-8)
    df.loc[lower_breakout, 'breakout_magnitude'] = (df['low_15d_min'] - df['low']) / (df['low_15d_min'] + 1e-8)
    
    # Gap Opening Analysis
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    
    # Gap Extremeness (percentile in 20-day distribution)
    df['gap_percentile'] = df['opening_gap'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) == 20 else np.nan)
    )
    
    # Gap-Volume Relationship
    df['gap_volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['gap_volume_support'] = np.sign(df['opening_gap']) * np.sign(df['gap_volume_ratio'] - 1)
    
    # Composite Alpha Generation
    # Momentum-Efficiency Integration
    momentum_efficiency = df['momentum_consistency'] * df['efficiency_persistence']
    momentum_efficiency_adj = momentum_efficiency * (1 + 0.5 * np.sign(df['efficiency_regime_shift']))
    
    # Volume-Context Weighting
    volume_weighted_momentum = momentum_efficiency_adj * (1 + df['volume_support_score'])
    volume_persistence_adj = volume_weighted_momentum * (1 + 0.3 * df['volume_persistence'])
    
    # Breakout Confirmation Layer
    breakout_enhanced = volume_persistence_adj * (1 + 0.5 * df['range_breakout'] * df['breakout_magnitude'])
    gap_enhanced = breakout_enhanced * (1 + 0.3 * df['gap_volume_support'] * df['gap_percentile'])
    
    # Final Signal Construction
    # Multi-timeframe confirmation
    multi_timeframe_confirmation = np.sign(df['momentum_3d']) * np.sign(df['momentum_8d'])
    
    # Final composite factor
    final_factor = gap_enhanced * (1 + 0.2 * multi_timeframe_confirmation)
    
    # Apply volume and breakout context filters
    volume_filter = np.where(df['volume_breakout_ratio'] > 0.1, 1, 0.5)
    breakout_filter = np.where(df['range_breakout'] != 0, 1.2, 1.0)
    
    result = final_factor * volume_filter * breakout_filter
    
    # Clean up and return
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
