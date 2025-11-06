import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Fractal Momentum Acceleration Patterns
    # Multi-Scale Velocity Divergence
    df['velocity_short'] = df['close'] / df['close'].shift(3) - 1
    df['velocity_medium'] = df['close'] / df['close'].shift(8) - 1
    df['velocity_long'] = df['close'] / df['close'].shift(21) - 1
    
    df['divergence_short_medium'] = df['velocity_short'] - df['velocity_medium']
    df['divergence_medium_long'] = df['velocity_medium'] - df['velocity_long']
    
    df['cross_timeframe_consistency'] = np.sign(df['divergence_short_medium']) * np.sign(df['divergence_medium_long'])
    
    # Fractal Acceleration Measurement
    df['acceleration_spread'] = np.abs(df['divergence_short_medium']) * np.abs(df['divergence_medium_long'])
    df['acceleration_reversal'] = df['divergence_short_medium'] * df['divergence_medium_long']
    
    # Volume-Enhanced Momentum Fractality
    df['volume_momentum'] = (df['volume'] / df['volume'].shift(5) - 1) - (df['volume'] / df['volume'].shift(10) - 1)
    df['volume_velocity_alignment'] = df['volume_momentum'] * df['divergence_short_medium']
    df['fractal_volume_consistency'] = np.sign(df['volume_momentum']) * np.sign(df['divergence_short_medium'])
    
    # Gap-Efficiency Microstructure Cascades
    df['gap_momentum'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Gap direction persistence
    gap_sign = np.sign(df['gap_momentum'])
    df['gap_persistence'] = 0
    for i in range(2, len(df)):
        if gap_sign.iloc[i] == gap_sign.iloc[i-1] == gap_sign.iloc[i-2]:
            df.iloc[i, df.columns.get_loc('gap_persistence')] = 1
    
    df['gap_magnitude_acceleration'] = df['gap_momentum'] - df['gap_momentum'].rolling(window=3, min_periods=1).mean()
    
    # Range Efficiency Momentum
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_momentum'] = df['intraday_efficiency'] - df['intraday_efficiency'].rolling(window=5, min_periods=1).mean()
    
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(np.abs(df['high'] - df['close'].shift(1)), 
                                     np.abs(df['low'] - df['close'].shift(1))))
    df['true_range_context'] = (df['high'] - df['low']) / true_range.rolling(window=8, min_periods=1).mean()
    
    # Microstructure Cascade Integration
    df['gap_efficiency_alignment'] = df['gap_momentum'] * df['efficiency_momentum']
    df['multi_timeframe_cascade'] = df['gap_efficiency_alignment'].rolling(window=3, min_periods=1).mean()
    df['cascade_persistence'] = np.sign(df['gap_efficiency_alignment']) * np.sign(df['gap_efficiency_alignment'].rolling(window=3, min_periods=1).mean())
    
    # Volume-Liquidity Flow Asymmetry
    df['volume_concentration'] = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    df['concentration_momentum'] = df['volume_concentration'] / df['volume_concentration'].rolling(window=5, min_periods=1).mean()
    df['concentration_divergence'] = (df['volume_concentration'].rolling(window=5, min_periods=1).mean() / 
                                     df['volume_concentration'].rolling(window=10, min_periods=1).mean())
    
    # Liquidity Flow Acceleration
    df['amount_flow'] = df['amount'] / df['amount'].shift(5) - 1
    df['volume_flow'] = df['volume'] / df['volume'].shift(5) - 1
    df['flow_acceleration'] = (df['volume'] / df['volume'].shift(5) - 1) - (df['volume'].shift(5) / df['volume'].shift(10) - 1)
    
    # Flow-Price Integration Asymmetry
    df['volume_price_alignment'] = df['volume_flow'] * df['velocity_short']
    df['amount_volume_divergence'] = df['amount_flow'] - df['volume_flow']
    df['flow_consistency'] = np.sign(df['volume_flow']) * np.sign(df['velocity_short'])
    
    # Fractal Regime Transition Dynamics
    df['daily_range'] = df['high'] - df['low']
    df['range_8day_avg'] = df['daily_range'].rolling(window=8, min_periods=1).mean()
    df['range_momentum'] = df['daily_range'] / df['range_8day_avg']
    
    # Regime classification
    df['expansion_regime'] = (df['daily_range'] > 1.5 * df['range_8day_avg']).astype(int)
    df['contraction_regime'] = (df['daily_range'] < 0.7 * df['range_8day_avg']).astype(int)
    df['normal_regime'] = ((df['expansion_regime'] == 0) & (df['contraction_regime'] == 0)).astype(int)
    
    # Microstructure Pattern Recognition
    df['anchor_deviation'] = np.abs(df['close'] - (df['high'] + df['low']) / 2) / (df['high'] - df['low']).replace(0, np.nan)
    df['anchor_breakout_strength'] = df['gap_momentum'] * df['anchor_deviation']
    df['multi_day_anchor_persistence'] = df['anchor_deviation'].rolling(window=3, min_periods=1).mean()
    
    # Volume-Price Integration
    volume_returns = []
    for i in range(len(df)):
        if i >= 4:
            vol_window = df['volume'].iloc[i-4:i+1]
            ret_window = np.abs(df['close'].iloc[i-4:i+1] / df['close'].iloc[i-5:i] - 1)
            if len(vol_window) == 5 and len(ret_window) == 5:
                corr = np.corrcoef(vol_window, ret_window)[0, 1] if not (np.std(vol_window) == 0 or np.std(ret_window) == 0) else 0
            else:
                corr = 0
        else:
            corr = 0
        volume_returns.append(corr)
    
    df['volume_return_correlation'] = volume_returns
    df['volume_entropy'] = df['volume_concentration'] * df['volume_return_correlation']
    df['integration_consistency'] = np.sign(df['volume_return_correlation']) * np.sign(df['velocity_short'])
    
    # Multi-Dimensional Fractal Factor Synthesis
    # Expansion regime components
    expansion_factor = (
        df['acceleration_spread'] * df['range_momentum'] *  # Enhanced fractal divergence
        df['gap_persistence'] * df['gap_magnitude_acceleration'] *  # Gap persistence emphasis
        df['volume_concentration'] * df['concentration_momentum']  # High-concentration volume
    )
    
    # Contraction regime components
    contraction_factor = (
        df['flow_consistency'] * df['efficiency_momentum'] *  # Dampened divergence
        df['flow_acceleration'] * df['amount_volume_divergence'] *  # Flow acceleration focus
        df['gap_efficiency_alignment'] * df['volume_price_alignment']  # Microstructure alignment
    )
    
    # Normal regime components
    normal_factor = (
        df['cross_timeframe_consistency'] *  # Multi-timeframe consistency
        df['volume_velocity_alignment'] *  # Volume-enhanced momentum
        df['fractal_volume_consistency'] *  # Fractal acceleration persistence
        df['anchor_breakout_strength'] *  # Anchor breakout confirmation
        df['integration_consistency']  # Volume-price integration
    )
    
    # Regime-adaptive combination
    result = (
        df['expansion_regime'] * expansion_factor +
        df['contraction_regime'] * contraction_factor +
        df['normal_regime'] * normal_factor
    )
    
    # Apply volume-liquidity flow scaling
    result = result * (1 + df['volume_flow']) * (1 + df['concentration_momentum'])
    
    # Apply microstructure pattern confirmation
    result = result * (1 + df['cascade_persistence']) * (1 + df['multi_day_anchor_persistence'])
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
