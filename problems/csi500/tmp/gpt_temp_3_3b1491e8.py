import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Divergence Factor
    """
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['daily_return'] = data['close'] / data['prev_close'] - 1
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-timeframe volatility regime classification
    # Short-term (3-day) volatility
    data['atr_3'] = data['true_range'].rolling(window=3, min_periods=3).mean()
    data['return_vol_3'] = data['daily_return'].rolling(window=3, min_periods=3).std()
    data['vol_composite_3'] = data['atr_3'] * data['return_vol_3']
    data['vol_regime_3'] = (data['vol_composite_3'] > data['vol_composite_3'].rolling(window=10, min_periods=10).median()).astype(int)
    
    # Medium-term (10-day) volatility
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=10).mean()
    data['return_vol_10'] = data['daily_return'].rolling(window=10, min_periods=10).std()
    data['vol_composite_10'] = data['atr_10'] * data['return_vol_10']
    data['vol_regime_10'] = (data['vol_composite_10'] > data['vol_composite_10'].rolling(window=20, min_periods=20).median()).astype(int)
    
    # Volatility regime transition analysis
    data['regime_change_3'] = (data['vol_regime_3'] != data['vol_regime_3'].shift(1)).astype(int)
    data['consecutive_regime_3'] = data.groupby((data['vol_regime_3'] != data['vol_regime_3'].shift(1)).cumsum()).cumcount() + 1
    data['regime_stability_3'] = 1 - (data['regime_change_3'].rolling(window=10, min_periods=10).sum() / 10)
    
    # Price momentum analysis
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['return_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_efficiency_3d'] = data['return_3d'] / data['atr_3']
    data['price_efficiency_10d'] = data['return_10d'] / data['atr_10']
    
    # Momentum divergence detection
    data['momentum_divergence_ultra_short'] = np.sign(data['daily_return']) - np.sign(data['return_3d'])
    data['momentum_divergence_short'] = np.sign(data['return_3d']) - np.sign(data['return_10d'])
    data['momentum_divergence_score'] = (abs(data['momentum_divergence_ultra_short']) + abs(data['momentum_divergence_short'])) / 2
    
    # Volume activity pattern recognition
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_intensity'] = data['volume'] / data['volume_5d_avg']
    data['volume_spike'] = (data['volume'] > 2.5 * data['volume'].rolling(window=10, min_periods=10).mean()).astype(int)
    
    data['trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['volume_slope_5d'] = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Volume regime classification
    data['volume_regime'] = (data['volume'] > data['volume'].rolling(window=10, min_periods=10).median()).astype(int)
    
    # Price-volume divergence calculation
    data['momentum_volume_corr_5d'] = data['daily_return'].rolling(window=5, min_periods=5).corr(data['volume_intensity'])
    data['efficiency_volume_divergence'] = data['price_efficiency_3d'] * data['volume_intensity']
    
    # Multi-scale divergence integration
    weights = {'ultra_short': 0.4, 'short': 0.35, 'medium': 0.25}
    data['overall_divergence'] = (
        weights['ultra_short'] * abs(data['momentum_divergence_ultra_short']) +
        weights['short'] * abs(data['momentum_divergence_short']) +
        weights['medium'] * abs(data['efficiency_volume_divergence'])
    )
    
    # Regime-adaptive signal enhancement
    # High volatility regime adjustments
    high_vol_mask = (data['vol_regime_3'] == 1)
    data['signal_amplification'] = 1.0
    data.loc[high_vol_mask, 'signal_amplification'] = 1.2  # Amplify short-term signals
    data.loc[~high_vol_mask, 'signal_amplification'] = 0.8  # Reduce sensitivity in low vol
    
    # Volume regime adjustments
    high_vol_mask = (data['volume_regime'] == 1)
    data['volume_confidence'] = 1.0
    data.loc[high_vol_mask, 'volume_confidence'] = 1.3  # Increase confidence in high volume
    data.loc[~high_vol_mask, 'volume_confidence'] = 0.7  # De-emphasize in low volume
    
    # Multi-regime signal fusion
    data['regime_combined_adjustment'] = (
        data['signal_amplification'] * 
        data['volume_confidence'] * 
        data['regime_stability_3']
    )
    
    # Multi-timeframe signal aggregation
    # Immediate signal (1-3 day horizon)
    data['immediate_signal'] = (
        data['momentum_divergence_ultra_short'] * 
        data['volume_spike'] * 
        data['regime_combined_adjustment']
    )
    
    # Short-term signal (3-10 day horizon)
    data['short_term_signal'] = (
        data['momentum_divergence_short'] * 
        data['momentum_volume_corr_5d'] * 
        data['regime_stability_3']
    )
    
    # Medium-term signal (10-20 day horizon)
    data['medium_term_signal'] = (
        data['efficiency_volume_divergence'] * 
        data['volume_slope_5d'] * 
        data['regime_stability_3']
    )
    
    # Time-weighted signal integration
    horizon_weights = {'immediate': 0.5, 'short_term': 0.3, 'medium_term': 0.2}
    data['combined_signal'] = (
        horizon_weights['immediate'] * data['immediate_signal'] +
        horizon_weights['short_term'] * data['short_term_signal'] +
        horizon_weights['medium_term'] * data['medium_term_signal']
    )
    
    # Dynamic scaling
    volatility_scaling = 1 / (1 + data['vol_composite_3'])
    volume_scaling = data['volume_confidence']
    regime_scaling = data['regime_stability_3']
    
    # Final factor generation
    data['factor'] = (
        data['combined_signal'] * 
        volatility_scaling * 
        volume_scaling * 
        regime_scaling * 
        data['overall_divergence']
    )
    
    # Clean up and return
    factor_series = data['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
