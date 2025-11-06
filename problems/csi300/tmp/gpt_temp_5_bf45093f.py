import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Adaptive Momentum Reversal with Volume Dynamics
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Calculate True Range and ATR
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['atr_20_median'] = data['atr_10'].rolling(window=20, min_periods=10).median()
    
    # Volatility regime detection
    data['high_vol_regime'] = data['atr_10'] > (1.5 * data['atr_20_median'])
    
    # Multi-Timeframe Momentum & Reversal Analysis
    # Short-term dynamics (1-5 days)
    data['ret_1'] = data['close'].pct_change(1)
    data['ret_3'] = data['close'].pct_change(3)
    data['ret_5'] = data['close'].pct_change(5)
    
    # Short-term reversal patterns
    data['reversal_1_3'] = -data['ret_1'] * data['ret_3']
    data['momentum_accel_3'] = data['ret_3'] - data['ret_1'].rolling(window=3).mean()
    
    # Medium-term dynamics (5-10 days)
    data['ret_10'] = data['close'].pct_change(10)
    data['ma_10'] = data['close'].rolling(window=10, min_periods=5).mean()
    data['price_dev_10'] = (data['close'] - data['ma_10']) / data['ma_10']
    
    # Medium-term reversal and acceleration
    data['reversal_accel_10'] = data['ret_10'] - data['ret_5']
    data['trend_strength_10'] = data['close'].rolling(window=10).apply(
        lambda x: (x[-1] - x[0]) / (np.std(x) + 1e-8) if len(x) == 10 else np.nan
    )
    
    # Cross-timeframe interaction
    data['momentum_decay'] = data['ret_5'] - data['ret_10']
    data['reversal_strength_diff'] = data['reversal_1_3'] - data['price_dev_10']
    data['accel_diff'] = data['momentum_accel_3'] - data['reversal_accel_10']
    
    # Volume-Price Dynamics Integration
    # Volume Trend Analysis
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_accel_5'] = data['volume_ma_5'].pct_change(3)
    data['volume_trend_strength'] = data['volume'].rolling(window=15).apply(
        lambda x: (x[-1] - x[0]) / (np.std(x) + 1e-8) if len(x) == 15 else np.nan
    )
    
    # Volume Anomaly Detection
    data['volume_deviation'] = (data['volume_ma_5'] - data['volume_ma_20']) / data['volume_ma_20']
    data['volume_anomaly'] = data['volume_deviation'].abs() > data['volume_deviation'].rolling(window=20).std()
    
    # Volume-Price Confirmation
    data['volume_price_divergence'] = data['ret_5'] * data['volume_accel_5']
    data['volume_signal_strength'] = data['volume_accel_5'] * data['volume_anomaly']
    
    # Regime-Adaptive Signal Construction
    # Volatility-Regime Parameter Selection
    data['regime_lookback'] = np.where(data['high_vol_regime'], 3, 10)
    data['regime_reversal_sensitivity'] = np.where(data['high_vol_regime'], 1.5, 1.0)
    
    # Multi-Scale Signal Weighting
    # High volatility: focus on short-term reversals
    high_vol_momentum = data['reversal_1_3'] * data['volume_signal_strength']
    high_vol_reversal = data['price_dev_10'] * data['regime_reversal_sensitivity']
    
    # Normal volatility: emphasize medium-term patterns
    normal_vol_momentum = data['trend_strength_10'] * data['volume_trend_strength']
    normal_vol_reversal = data['price_dev_10'] * 0.7  # Conservative reversal
    
    # Dynamic adjustment based on regime classification
    data['momentum_component'] = np.where(
        data['high_vol_regime'], 
        high_vol_momentum, 
        normal_vol_momentum
    )
    
    data['reversal_component'] = np.where(
        data['high_vol_regime'], 
        high_vol_reversal, 
        normal_vol_reversal
    )
    
    # Volume-Enhanced Reversal Signals
    data['volume_confirmed_reversal'] = data['reversal_component'] * (
        1 + data['volume_price_divergence'] * data['volume_anomaly']
    )
    
    # Regime-dependent mean reversion
    data['mean_reversion_signal'] = np.where(
        data['high_vol_regime'],
        -data['ret_3'] * data['volume_accel_5'],  # 3-day for high volatility
        -data['ret_10'] * data['volume_trend_strength']  # 10-day for normal volatility
    )
    
    # Integrated Factor Generation
    # Volatility-Adjusted Momentum Component
    volatility_scaling = np.where(data['high_vol_regime'], 0.7, 1.2)
    data['vol_adj_momentum'] = (
        data['momentum_component'] * volatility_scaling + 
        data['momentum_decay'] * 0.3
    )
    
    # Volume-Confirmed Reversal Component
    data['volume_confirmed_component'] = (
        data['volume_confirmed_reversal'] + 
        data['mean_reversion_signal'] * data['volume_signal_strength']
    )
    
    # Final Predictive Signal
    # Combined momentum-reversal interaction
    short_term_weight = np.where(data['high_vol_regime'], 0.6, 0.3)
    medium_term_weight = np.where(data['high_vol_regime'], 0.4, 0.7)
    
    final_signal = (
        short_term_weight * data['vol_adj_momentum'] +
        medium_term_weight * data['volume_confirmed_component'] +
        data['accel_diff'] * 0.2 +
        data['reversal_strength_diff'] * 0.15
    )
    
    # Clean up intermediate columns
    result = final_signal.copy()
    result.name = 'multi_scale_vol_adaptive_factor'
    
    return result
