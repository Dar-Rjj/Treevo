import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Momentum Regime Adaptive Alpha with Microstructure Efficiency
    """
    data = df.copy()
    
    # Volatility-Momentum Regime Identification
    # Volatility Component
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Rolling volatility percentiles
    data['vol_percentile'] = data['true_range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2)) if x.quantile(0.8) != x.quantile(0.2) else 0.5
    )
    
    # Momentum Component
    data['short_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['medium_momentum'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['momentum_acceleration'] = data['short_momentum'] / np.where(data['medium_momentum'] != 0, data['medium_momentum'], 1)
    
    # Regime-Adaptive Signal Generation
    # High Volatility & Strong Momentum Regime
    data['momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    
    # Price extremes detection (t-9 to t)
    data['highest_high_10d'] = data['high'].rolling(window=10, min_periods=5).max()
    data['lowest_low_10d'] = data['low'].rolling(window=10, min_periods=5).min()
    
    data['upper_bound_dist'] = (data['highest_high_10d'] - data['close']) / data['highest_high_10d']
    data['lower_bound_dist'] = (data['close'] - data['lowest_low_10d']) / data['close']
    data['high_vol_signal'] = data['momentum_2d'] * (data['upper_bound_dist'] - data['lower_bound_dist'])
    
    # Low Volatility & Reversal Regime
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_pressure'] = (data['close'] - data['open']) / np.where(data['high'] - data['low'] != 0, data['high'] - data['low'], 1)
    data['low_vol_signal'] = data['momentum_5d'] * data['price_pressure']
    
    # Normal Volatility & Transition Regime
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['volume_efficiency'] = data['amount'] / (data['volume'] * np.where(data['high'] - data['low'] != 0, data['high'] - data['low'], 1))
    data['normal_vol_signal'] = data['momentum_3d'] * data['volume_efficiency']
    
    # Microstructure Efficiency Analysis
    # Order Flow Efficiency
    data['trade_size_indicator'] = data['amount'] / np.where(data['volume'] != 0, data['volume'], 1)
    data['price_impact_efficiency'] = (data['close'] - data['close'].shift(1)) / np.where(data['volume'] != 0, data['volume'], 1)
    data['volume_price_coherence'] = np.sign(data['close'] - data['close'].shift(1)) * data['volume']
    
    # Volume Momentum Integration
    data['volume_momentum_5d'] = (data['volume'] - data['volume'].shift(5)) / np.where(data['volume'].shift(5) != 0, data['volume'].shift(5), 1)
    data['volume_momentum_20d'] = (data['volume'] - data['volume'].shift(20)) / np.where(data['volume'].shift(20) != 0, data['volume'].shift(20), 1)
    data['volume_acceleration'] = data['volume_momentum_5d'] / np.where(data['volume_momentum_20d'] != 0, data['volume_momentum_20d'], 1)
    data['volume_confirmation'] = data['volume_acceleration'] * data['volume_price_coherence']
    
    # Adaptive Factor Integration
    # Regime-Weighted Signal Selection
    data['volatility_weight'] = data['vol_percentile']
    data['momentum_weight'] = 1 / (1 + np.abs(data['momentum_acceleration']))
    
    # Combined regime signal
    data['combined_regime_signal'] = (
        data['volatility_weight'] * data['momentum_weight'] * 
        np.where(data['vol_percentile'] > 0.6, data['high_vol_signal'],
                np.where(data['vol_percentile'] < 0.4, data['low_vol_signal'],
                        data['normal_vol_signal']))
    )
    
    # Microstructure Confirmation
    data['efficiency_multiplier'] = 1 + (data['trade_size_indicator'] * data['price_impact_efficiency'])
    
    # Volume confirmation strength
    data['volume_confirmation_abs_5d_avg'] = np.abs(data['volume_confirmation']).rolling(window=5, min_periods=3).mean()
    data['volume_confirmation_strength'] = data['volume_confirmation'] / np.where(data['volume_confirmation_abs_5d_avg'] != 0, data['volume_confirmation_abs_5d_avg'], 1)
    
    # Final Alpha Factor
    data['core_signal'] = data['combined_regime_signal'] * data['efficiency_multiplier']
    data['final_alpha'] = data['core_signal'] * data['volume_confirmation_strength']
    
    return data['final_alpha']
