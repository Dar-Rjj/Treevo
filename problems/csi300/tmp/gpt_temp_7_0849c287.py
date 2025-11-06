import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Efficiency Momentum with Volume-Price Microstructure Alignment
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Multi-Timeframe Efficiency Momentum
    # Daily price efficiency
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Ultra-Short Efficiency (1-3 day)
    data['efficiency_3d_avg'] = data['daily_efficiency'].rolling(window=3, min_periods=1).mean()
    data['ultra_short_momentum'] = data['daily_efficiency'] - data['efficiency_3d_avg']
    data['efficiency_acceleration'] = data['daily_efficiency'] - data['daily_efficiency'].shift(3)
    
    # Short-Term Efficiency (5-8 day)
    data['efficiency_5d_avg'] = data['daily_efficiency'].rolling(window=5, min_periods=1).mean()
    data['efficiency_8d_std'] = data['daily_efficiency'].rolling(window=8, min_periods=1).std()
    data['short_term_trend'] = data['daily_efficiency'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    
    # Medium-Term Efficiency (10-15 day)
    data['efficiency_15d_vol'] = data['daily_efficiency'].rolling(window=15, min_periods=1).std()
    data['efficiency_mean_reversion'] = (
        data['daily_efficiency'] - data['daily_efficiency'].rolling(window=15, min_periods=1).mean()
    ) / (data['efficiency_15d_vol'] + 1e-8)
    data['efficiency_persistence'] = data['daily_efficiency'].rolling(window=10, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if (x[i] - x[i-1]) * (x[0] - x[-1]) > 0]) / max(len(x)-1, 1)
    )
    
    # Efficiency Momentum Divergence
    data['ultra_vs_short_div'] = data['ultra_short_momentum'] - (
        data['daily_efficiency'] - data['efficiency_5d_avg']
    )
    data['short_vs_medium_div'] = (
        data['daily_efficiency'] - data['efficiency_5d_avg']
    ) - data['efficiency_mean_reversion']
    data['ultra_vs_medium_div'] = data['ultra_short_momentum'] - data['efficiency_mean_reversion']
    
    # 2. Volume-Price Microstructure Alignment
    # Volume Efficiency Analysis
    data['vw_efficiency'] = data['daily_efficiency'] * data['volume']
    data['volume_direction_consistency'] = (
        np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    ).rolling(window=5, min_periods=1).mean()
    data['volume_pressure_efficiency'] = data['daily_efficiency'] * (
        data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    )
    
    # Volume Acceleration Signals
    data['volume_5d_momentum'] = data['volume'] / (data['volume'].shift(5) + 1e-8) - 1
    data['volume_ratio'] = (
        data['volume'].rolling(window=5, min_periods=1).mean() / 
        data['volume'].rolling(window=20, min_periods=1).mean()
    )
    data['volume_breakout'] = (
        data['volume'] > 1.5 * data['volume'].rolling(window=20, min_periods=1).mean()
    ).astype(float)
    
    # Micro-Structure Price Action
    data['ohlc_efficiency_ratio'] = (
        (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    )
    data['failed_breakout'] = (
        (data['high'] == data['close']) & (data['close'] < data['high'].shift(1))
    ).astype(float) - (
        (data['low'] == data['close']) & (data['close'] > data['low'].shift(1))
    ).astype(float)
    data['intraday_reversal_efficiency'] = (
        (data['close'] - data['open']) * (data['open'] - data['close'].shift(1))
    ) / (data['high'] - data['low'] + 1e-8)
    
    # Volume-Weighted Micro-Structure
    data['vw_high_efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8) * data['volume']
    data['vw_low_efficiency'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8) * data['volume']
    data['trade_size_impact'] = (data['amount'] / (data['volume'] + 1e-8)) * data['daily_efficiency']
    data['microstructure_absorption'] = (
        data['vw_high_efficiency'] - data['vw_low_efficiency']
    ) / (data['volume'] + 1e-8)
    
    # 3. Volatility-Regime Context
    # Efficiency Volatility Structure
    data['efficiency_vol_5d'] = data['daily_efficiency'].rolling(window=5, min_periods=1).std()
    data['efficiency_vol_15d'] = data['daily_efficiency'].rolling(window=15, min_periods=1).std()
    data['efficiency_vol_ratio'] = data['efficiency_vol_5d'] / (data['efficiency_vol_15d'] + 1e-8)
    
    # Volatility Regime Classification
    data['high_vol_regime'] = (data['efficiency_vol_5d'] > data['efficiency_vol_5d'].rolling(window=20, min_periods=1).quantile(0.7)).astype(float)
    data['vol_regime_transition'] = data['high_vol_regime'].diff()
    data['vol_persistence'] = data['high_vol_regime'].rolling(window=5, min_periods=1).mean()
    
    # Volatility Context Adjustment
    data['scaled_efficiency_momentum'] = data['ultra_short_momentum'] / (data['efficiency_vol_5d'] + 1e-8)
    data['adjusted_volume_confirmation'] = data['volume_direction_consistency'] * (1 - data['high_vol_regime'])
    data['regime_amplification'] = 1 + data['vol_regime_transition'].abs() * 0.5
    
    # 4. Liquidity and Microstructure Filters
    # Liquidity Efficiency Metrics
    data['amount_efficiency'] = data['amount'] * data['daily_efficiency']
    data['volume_consistency'] = data['volume'] / (data['volume'].rolling(window=15, min_periods=1).std() + 1e-8)
    data['trade_size_efficiency'] = (data['amount'] / (data['volume'] + 1e-8)) * data['daily_efficiency']
    
    # Microstructure Confirmation
    data['micro_alignment_filter'] = (
        data['volume_direction_consistency'] * data['trade_size_efficiency']
    ).abs()
    data['volume_efficiency_correlation'] = (
        data['daily_efficiency'].rolling(window=5, min_periods=1).corr(data['volume'])
    )
    data['failed_breakout_confirmation'] = data['failed_breakout'] * data['intraday_reversal_efficiency']
    
    # Liquidity-Weighted Signals
    data['liquidity_validation'] = data['amount_efficiency'] * data['volume_consistency']
    data['micro_timing_filter'] = (
        data['microstructure_absorption'] * data['volume_breakout']
    )
    data['liquidity_micro_adjustment'] = (
        data['liquidity_validation'] * data['micro_timing_filter']
    ) / (data['efficiency_vol_5d'] + 1e-8)
    
    # 5. Synthesize Composite Alpha Factor
    # Combine Efficiency Momentum with Volume Alignment
    data['efficiency_volume_asymmetry'] = (
        data['ultra_vs_short_div'] * data['volume_direction_consistency'] +
        data['short_vs_medium_div'] * data['volume_5d_momentum']
    )
    data['micro_validated_momentum'] = (
        data['efficiency_volume_asymmetry'] * data['micro_alignment_filter']
    )
    
    # Multi-Dimensional Context Adjustments
    data['vol_scaled_component'] = (
        data['scaled_efficiency_momentum'] * (1 + data['efficiency_vol_ratio'])
    )
    data['liquidity_confirmed_component'] = (
        data['liquidity_micro_adjustment'] * data['volume_efficiency_correlation']
    )
    data['cross_timeframe_validation'] = (
        data['ultra_vs_medium_div'] * data['efficiency_persistence']
    )
    
    # Final Alpha Integration
    data['composite_alpha'] = (
        data['micro_validated_momentum'] * 0.4 +
        data['vol_scaled_component'] * 0.3 +
        data['liquidity_confirmed_component'] * 0.2 +
        data['cross_timeframe_validation'] * 0.1
    ) * data['regime_amplification']
    
    # Apply regime-dependent efficiency weighting
    data['regime_weight'] = np.where(
        data['high_vol_regime'] == 1,
        0.7 + 0.3 * data['vol_persistence'],
        1.0 - 0.2 * data['vol_persistence']
    )
    
    # Final microstructure-validated efficiency momentum factor
    data['final_alpha'] = data['composite_alpha'] * data['regime_weight']
    
    # Clean up and return
    result = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
