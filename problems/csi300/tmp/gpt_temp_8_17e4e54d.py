import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Gap Analysis
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_efficiency'] = np.abs(data['gap']) / (data['high'] - data['low'])
    data['gap_2d_momentum'] = data['gap'] / data['gap'].rolling(window=2).mean() - 1
    data['gap_5d_momentum'] = data['gap'] / data['gap'].rolling(window=5).mean() - 1
    
    # Gap direction persistence
    gap_sign = np.sign(data['gap'])
    gap_sign_shifted = gap_sign.shift(1)
    data['gap_persistence'] = gap_sign.rolling(window=5).apply(
        lambda x: sum(x[i] == x[i-1] for i in range(1, len(x)) if not pd.isna(x[i]) and not pd.isna(x[i-1])), 
        raw=False
    )
    
    # Efficiency-Volume Integration
    data['efficiency_ratio'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_momentum'] = data['volume'] / data['volume'].rolling(window=5).mean() - 1
    data['eff_vol_corr'] = data['efficiency_ratio'].rolling(window=10).corr(data['volume_momentum'])
    data['volume_weighted_gap'] = data['gap'] * (data['volume'] / data['volume'].rolling(window=20).mean())
    data['volume_weighted_efficiency'] = data['efficiency_ratio'] * data['volume_momentum']
    
    # Volatility-Regime Framework
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_ratio'] = data['true_range'].rolling(window=5).mean() / data['true_range'].rolling(window=20).mean()
    data['high_vol_regime'] = data['volatility_ratio'] > 1.5
    data['low_vol_regime'] = data['volatility_ratio'] < 0.8
    data['vol_scaled_gap_momentum'] = data['gap'] * data['volatility_ratio']
    
    # Liquidity Confirmation System
    data['liquidity'] = data['amount'] / data['volume']
    data['liquidity_momentum'] = data['liquidity'] / data['liquidity'].rolling(window=10).mean() - 1
    data['liquidity_pressure'] = (data['high'] + data['low']) / 2 - data['close']
    
    # Volume trend strength
    volume_shifted = data['volume'].shift(1)
    data['volume_trend_strength'] = data['volume'].rolling(window=5).apply(
        lambda x: sum(x[i] > x[i-1] for i in range(1, len(x)) if not pd.isna(x[i]) and not pd.isna(x[i-1])), 
        raw=False
    )
    data['liquidity_confirmed_gap'] = data['gap'] * data['liquidity_momentum']
    
    # Intraday Pressure Dynamics
    data['morning_impact'] = (data['high'] - data['open']) / data['true_range']
    data['afternoon_impact'] = (data['close'] - data['low']) / data['true_range']
    data['session_asymmetry'] = data['morning_impact'] - data['afternoon_impact']
    data['intraday_pressure'] = (np.maximum(0, data['close'] - data['open']) - np.maximum(0, data['open'] - data['close'])) * data['volume']
    data['pressure_confirmed_gap'] = data['gap'] * data['intraday_pressure']
    
    # Acceleration and Momentum Synthesis
    data['gap_acceleration'] = data['gap_2d_momentum'] - data['gap_5d_momentum']
    eff_ratio_2d = data['efficiency_ratio'] / data['efficiency_ratio'].rolling(window=2).mean()
    eff_ratio_5d = data['efficiency_ratio'] / data['efficiency_ratio'].rolling(window=5).mean()
    data['efficiency_acceleration'] = eff_ratio_2d - eff_ratio_5d
    data['gap_breakout_efficiency'] = data['gap_acceleration'] * (data['true_range'] / data['true_range'].rolling(window=3).mean())
    data['volume_confirmed_acceleration'] = data['gap_acceleration'] * data['volume_momentum']
    
    # Regime-Adaptive Signal Processing
    # High Volatility Factors
    data['vol_scaled_gap_acceleration'] = data['gap_acceleration'] * data['volatility_ratio']
    data['extreme_gap_reversal'] = np.where(
        data['gap_efficiency'] > 0.8, 
        -data['gap'], 
        data['gap']
    )
    
    # Low Volatility Factors
    data['gap_breakout_anticipation'] = data['gap_efficiency'] / data['gap_efficiency'].rolling(window=20).max()
    data['range_expansion_gap'] = data['gap'] * (1 / data['volatility_ratio'])
    
    # Liquidity-Enhanced Factors
    data['liquidity_confirmed_gap_acceleration'] = data['gap_acceleration'] * data['liquidity_momentum']
    data['efficiency_liquidity_alignment'] = data['efficiency_acceleration'] * data['liquidity_momentum']
    
    # Composite Alpha Integration
    # Core Gap Momentum Construction
    core_gap_momentum = (
        data['gap'] * 0.3 +
        data['gap_2d_momentum'] * 0.25 +
        data['gap_5d_momentum'] * 0.2 +
        data['gap_efficiency'] * 0.25
    ) * (1 + data['volume_momentum'] * 0.1 + data['liquidity_momentum'] * 0.1)
    
    # Volatility-Regime Adaptation
    regime_adaptive_factor = np.where(
        data['high_vol_regime'],
        data['vol_scaled_gap_acceleration'] * 0.6 + data['extreme_gap_reversal'] * 0.4,
        np.where(
            data['low_vol_regime'],
            data['gap_breakout_anticipation'] * 0.5 + data['range_expansion_gap'] * 0.5,
            data['gap_acceleration'] * 0.7 + data['gap_efficiency'] * 0.3
        )
    )
    
    # Intraday Confirmation Layer
    intraday_confirmation = (
        data['session_asymmetry'] * 0.3 +
        data['intraday_pressure'] * 0.4 +
        data['pressure_confirmed_gap'] * 0.3
    )
    
    # Final Signal Generation
    final_signal = (
        core_gap_momentum * 0.4 +
        regime_adaptive_factor * 0.35 +
        intraday_confirmation * 0.25
    )
    
    # Apply liquidity and volume confirmation filters
    liquidity_filter = np.where(
        (data['liquidity_momentum'] > 0) & (data['volume_trend_strength'] >= 3),
        1.2,  # Boost signal with strong liquidity
        np.where(
            (data['liquidity_momentum'] < -0.1) | (data['volume_trend_strength'] <= 1),
            0.8,  # Reduce signal with weak liquidity
            1.0   # Neutral
        )
    )
    
    enhanced_signal = final_signal * liquidity_filter
    
    return enhanced_signal
