import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price differences and ratios
    df['close_diff_1'] = df['close'].diff(1)
    df['close_diff_2'] = df['close'].diff(2)
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_range'] = abs(df['close'] - df['open'])
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['trade_size'] = df['amount'] / df['volume']
    
    # Gap Fracture Intensity
    df['gap_fracture'] = abs(df['open'] - df['close'].shift(1)) / (abs(df['close'].shift(1) - df['close'].shift(2)) + 1e-8)
    
    # Intraday Fracture Magnitude
    df['intraday_fracture'] = df['high_low_range'] / (df['close_open_range'] + 1e-8)
    
    # Momentum Disruption
    df['momentum_disruption'] = abs(df['close_diff_1'] - df['close_diff_1'].shift(1)) / (abs(df['close_diff_1'].shift(1)) + 1e-8)
    
    # Volume Shock Intensity
    df['volume_shock'] = (df['volume'] / df['volume'].shift(1)) - (df['volume'].shift(1) / df['volume'].shift(2))
    
    # Trade Size Discontinuity
    df['trade_size_discontinuity'] = abs(df['trade_size'] - df['trade_size'].shift(1)) / (df['trade_size'].shift(1) + 1e-8)
    
    # Volume-Price Fracture
    df['volume_price_fracture'] = abs(df['volume_ratio'] - (df['close_diff_1'] / (df['close_diff_1'].shift(1) + 1e-8)))
    
    # Volatility Fracture
    df['volatility_fracture'] = (df['high_low_range'] / (df['high_low_range'].shift(1) + 1e-8)) - (df['high_low_range'].shift(1) / (df['high_low_range'].shift(2) + 1e-8))
    
    # Range Expansion Break
    df['range_expansion'] = df['high_low_range'] / ((df['high_low_range'].shift(1) + df['high_low_range'].shift(2)) / 2 + 1e-8)
    
    # Volatility Cascade (5-day rolling)
    df['volatility_cascade'] = df['high_low_range'].rolling(window=5).apply(
        lambda x: sum(x.iloc[i] > 1.5 * x.iloc[i-1] for i in range(1, 5)) / 5 if len(x) == 5 else np.nan
    )
    
    # Volume Regime Fracture
    df['volume_regime_fracture'] = (df['volume'] / ((df['volume'].shift(2) + df['volume'].shift(3) + df['volume'].shift(4)) / 3 + 1e-8)) - \
                                  (df['volume'].shift(1) / ((df['volume'].shift(3) + df['volume'].shift(4) + df['volume'].shift(5)) / 3 + 1e-8))
    
    # Trade Size Break
    df['trade_size_break'] = df['trade_size'] / ((df['trade_size'].shift(2) + df['trade_size'].shift(3) + df['trade_size'].shift(4)) / 3 + 1e-8)
    
    # Volume-Volatility Break
    df['volume_volatility_break'] = np.sign(df['volume_ratio'] - 1) * np.sign(df['high_low_range'] / (df['high_low_range'].shift(1) + 1e-8) - 1)
    
    # Correlation Fracture
    df['correlation_fracture'] = (df['close_diff_1'] * df['volume']) - (df['close_diff_1'].shift(1) * df['volume'].shift(1))
    
    # Range-Volume Dislocation
    df['range_volume_dislocation'] = (df['high_low_range'] * df['volume']) - (df['high_low_range'].shift(1) * df['volume'].shift(1))
    
    # Efficiency Fracture
    df['efficiency_fracture'] = (df['close_open_range'] / (df['high_low_range'] + 1e-8)) - \
                               (df['close_open_range'].shift(1) / (df['high_low_range'].shift(1) + 1e-8))
    
    # Reversal Fracture Density (5-day rolling)
    df['reversal_fracture'] = df['close_diff_1'].rolling(window=5).apply(
        lambda x: sum(np.sign(x.iloc[i]) != np.sign(x.iloc[i-2]) for i in range(2, 5)) / 5 if len(x) == 5 else np.nan
    )
    
    # Momentum Fracture Persistence (5-day rolling)
    df['momentum_fracture_persistence'] = df['close_diff_1'].rolling(window=5).apply(
        lambda x: sum(abs(x.iloc[i] / (x.iloc[i-1] + 1e-8)) > 2 for i in range(1, 5)) / 5 if len(x) == 5 else np.nan
    )
    
    # Upper and Lower Fracture Intensity
    df['upper_fracture'] = (df['high'] - df['close']) / ((df['close'] - df['low']) + 1e-8)
    df['lower_fracture'] = (df['close'] - df['low']) / ((df['high'] - df['close']) + 1e-8)
    df['net_fracture_pressure'] = (df['high'] - df['close']) - (df['close'] - df['low'])
    
    # Volume-Weighted Fracture
    df['upside_fracture_volume'] = df['volume'] * (df['high'] - df['close']) / (df['high_low_range'] + 1e-8)
    df['downside_fracture_volume'] = df['volume'] * (df['close'] - df['low']) / (df['high_low_range'] + 1e-8)
    df['net_fracture_imbalance'] = df['upside_fracture_volume'] - df['downside_fracture_volume']
    
    # Liquidity Asymmetry Factor
    df['liquidity_asymmetry'] = df['net_fracture_pressure'] * df['net_fracture_imbalance']
    
    # Fracture Transition Factor
    df['volatility_transition'] = (df['high_low_range'] / (df['high_low_range'].shift(1) + 1e-8)) * (df['close_diff_1'] / (abs(df['close_diff_1'].shift(1)) + 1e-8))
    df['volume_transition'] = df['volume_ratio'] * (df['close_diff_1'] / (abs(df['close_diff_1'].shift(1)) + 1e-8))
    df['fracture_transition'] = df['volatility_transition'] * df['volume_transition']
    
    # Market Impact Fracture Factor
    df['large_trade_fracture'] = df['trade_size'] * (df['high_low_range'] / (df['close_open_range'] + 1e-8))
    df['trade_size_fracture_ratio'] = (df['trade_size'] / (df['trade_size'].shift(1) + 1e-8)) * (df['close_open_range'] / (df['close_open_range'].shift(1) + 1e-8))
    df['market_impact_fracture'] = df['large_trade_fracture'] * df['trade_size_fracture_ratio']
    
    # Fractal Break Factor
    df['multi_scale_fracture'] = (df['close_diff_1'] / (abs(df['close_diff_1'].shift(1)) + 1e-8)) * \
                                (df['close'].diff(3) / (abs(df['close'].diff(3).shift(1)) + 1e-8))
    df['volume_range_fracture'] = df['volume_ratio'] * (df['high_low_range'] / (df['high_low_range'].shift(1) + 1e-8))
    df['fractal_break'] = df['multi_scale_fracture'] * df['volume_range_fracture']
    
    # Validation signals
    df['opening_closing_fracture'] = np.sign((df['open'] - df['low']) - (df['high'] - df['open'])) * \
                                    np.sign((df['close'] - df['low']) - (df['high'] - df['close']))
    
    # Liquidity Flow Persistence (5-day rolling)
    df['session_liquidity_flow'] = ((df['open'] - df['low']) - (df['high'] - df['open'])) * \
                                  ((df['close'] - df['low']) - (df['high'] - df['close']))
    df['liquidity_persistence'] = df['session_liquidity_flow'].rolling(window=5).apply(
        lambda x: sum(np.sign(x.iloc[i]) == np.sign(x.iloc[i-1]) for i in range(1, 5)) / 5 if len(x) == 5 else np.nan
    )
    
    # Volume Fracture Persistence (5-day rolling)
    df['volume_fracture_persistence'] = df['volume_ratio'].rolling(window=5).apply(
        lambda x: sum(abs(x.iloc[i] - 1) > 0.5 for i in range(1, 5)) / 5 if len(x) == 5 else np.nan
    )
    
    # Efficiency-Break Alignment
    df['efficiency_break_alignment'] = np.sign(df['efficiency_fracture']) * np.sign(df['close_diff_1'])
    
    # Volume-Fracture Confirmation
    df['volume_fracture_confirmation'] = np.sign(df['volume_ratio'] - 1) * np.sign(df['close_diff_1'])
    
    # Boundary Breakout Validation
    df['boundary_breakout_validation'] = df['volatility_transition'] * df['volume_transition']
    
    # Impact-Velocity Coherence
    df['acceleration_fracture'] = abs((df['close']/df['close'].shift(1) - 1) - (df['close'].shift(1)/df['close'].shift(2) - 1)) / \
                                 (abs(df['close'].shift(1)/df['close'].shift(2) - 1) + 1e-8)
    df['impact_fracture_velocity'] = df['large_trade_fracture'] * df['acceleration_fracture']
    df['impact_velocity_coherence'] = df['market_impact_fracture'] * df['impact_fracture_velocity']
    
    # Efficiency Fracture Validation
    df['efficiency_fracture_validation'] = (df['close_open_range'] / (df['high_low_range'] + 1e-8)) * \
                                          (df['volume_shock'] + df['trade_size_discontinuity'] + df['volume_price_fracture'])
    
    # Weighting scheme
    df['low_fracture_weight'] = 1 / (1 + df['gap_fracture'])
    df['fracture_stability_weight'] = 1 - abs(df['volatility_fracture'] - 1)
    df['convergence_confidence'] = (df['efficiency_break_alignment'] + df['volume_fracture_confirmation']) / 2
    df['persistence_confidence'] = (df['liquidity_persistence'] + df['volume_fracture_persistence']) / 2
    
    # Final alpha components
    df['converged_liquidity_flow'] = df['liquidity_asymmetry'] * df['opening_closing_fracture']
    df['validated_fracture_breakout'] = df['fracture_transition'] * df['boundary_breakout_validation']
    df['confirmed_impact_fracture'] = df['market_impact_fracture'] * df['impact_velocity_coherence']
    df['aligned_fractal_break'] = df['fractal_break'] * df['efficiency_fracture_validation']
    
    # Composite Fracture Alpha
    alpha = (df['converged_liquidity_flow'] * df['low_fracture_weight'] +
             df['validated_fracture_breakout'] * df['fracture_stability_weight'] +
             df['confirmed_impact_fracture'] * df['convergence_confidence'] +
             df['aligned_fractal_break'] * df['persistence_confidence'])
    
    return alpha
