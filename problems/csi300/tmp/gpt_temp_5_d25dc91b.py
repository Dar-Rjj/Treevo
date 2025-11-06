import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility-adjusted price-volume divergence,
    range compression breakout, efficiency-momentum transitions, and adaptive range-volume efficiency.
    """
    data = df.copy()
    
    # Volatility-Adjusted Price-Volume Divergence Component
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling volatility measures
    data['vol_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['vol_60d_q25'] = data['vol_10d'].rolling(window=60, min_periods=30).quantile(0.25)
    data['vol_60d_q75'] = data['vol_10d'].rolling(window=60, min_periods=30).quantile(0.75)
    
    # Volatility regime classification
    conditions = [
        data['vol_10d'] > data['vol_60d_q75'],
        data['vol_10d'] < data['vol_60d_q25'],
        True  # medium volatility
    ]
    choices = [2, 0, 1]  # 2: high, 1: medium, 0: low
    data['vol_regime'] = np.select(conditions, choices, default=1)
    
    # Price momentum calculations
    data['price_ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_accel_3d'] = (data['close'] / data['close'].shift(3) - 1) - (data['close'].shift(3) / data['close'].shift(6) - 1)
    
    # Volume momentum calculations
    data['volume_ret_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_accel_3d'] = (data['volume'] / data['volume'].shift(3) - 1) - (data['volume'].shift(3) / data['volume'].shift(6) - 1)
    
    # Price-volume divergence
    data['pv_divergence'] = np.sign(data['price_ret_5d']) * np.sign(data['volume_ret_5d'])
    data['pv_divergence_mag'] = abs(data['price_ret_5d'] - data['volume_ret_5d'])
    
    # Regime-adaptive divergence signal
    high_vol_signal = -data['pv_divergence'] * data['pv_divergence_mag'] * data['vol_10d']
    medium_vol_signal = data['price_accel_3d'] * (1 + data['pv_divergence_mag'])
    low_vol_signal = data['pv_divergence'] * data['pv_divergence_mag'] * data['price_ret_5d']
    
    conditions_pv = [
        data['vol_regime'] == 2,
        data['vol_regime'] == 0,
        True  # medium volatility
    ]
    choices_pv = [high_vol_signal, low_vol_signal, medium_vol_signal]
    data['pv_signal'] = np.select(conditions_pv, choices_pv, default=medium_vol_signal)
    
    # Range Compression Breakout Component
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_5d_std'] = data['daily_range'].rolling(window=5, min_periods=3).std()
    
    # Compression detection
    data['range_compression'] = (data['daily_range'] < data['range_5d_avg'] - 0.5 * data['range_5d_std']).astype(int)
    data['compression_streak'] = data['range_compression'].rolling(window=5, min_periods=1).sum()
    
    # Amount per Volume analysis
    data['apv'] = data['amount'] / data['volume']
    data['apv_momentum'] = data['apv'] / data['apv'].shift(5) - 1
    
    # Breakout signal
    data['range_expansion'] = (data['daily_range'] > data['range_5d_avg'] + data['range_5d_std']).astype(int)
    data['apv_surge'] = (data['apv_momentum'] > data['apv_momentum'].rolling(window=10, min_periods=5).quantile(0.7)).astype(int)
    
    data['breakout_signal'] = data['range_expansion'] * data['apv_surge'] * np.sign(data['close'] - data['open'])
    
    # Efficiency-Momentum Component
    data['net_movement'] = abs(data['close'] - data['close'].shift(5))
    data['total_range'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    data['efficiency_ratio'] = data['net_movement'] / data['total_range']
    
    # Efficiency regime
    eff_conditions = [
        data['efficiency_ratio'] > 0.7,
        data['efficiency_ratio'] < 0.3,
        True
    ]
    eff_choices = [2, 0, 1]  # 2: high, 1: medium, 0: low efficiency
    data['efficiency_regime'] = np.select(eff_conditions, eff_choices, default=1)
    
    # Multi-timeframe momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_8d']
    
    # Volume regime
    data['volume_avg_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    vol_regime_conditions = [
        data['volume'] > 1.5 * data['volume_avg_10d'],
        data['volume'] < 0.7 * data['volume_avg_10d'],
        True
    ]
    vol_regime_choices = [2, 0, 1]  # 2: high, 1: normal, 0: low
    data['volume_regime'] = np.select(vol_regime_conditions, vol_regime_choices, default=1)
    
    # Efficiency-momentum signal
    data['eff_mom_signal'] = data['momentum_divergence'] * data['efficiency_ratio'] * (data['volume_regime'] / 2)
    
    # Adaptive Range-Volume Efficiency Component
    data['range_3d_avg'] = data['daily_range'].rolling(window=3, min_periods=2).mean()
    data['range_percentile_10d'] = data['daily_range'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Range efficiency
    data['range_efficiency'] = data['price_ret_5d'] / (data['range_3d_avg'] + 1e-8)
    
    # Volume efficiency
    data['volume_per_range'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['volume_efficiency'] = data['volume_per_range'] / data['volume_per_range'].rolling(window=10, min_periods=5).mean()
    
    # Efficiency divergence
    data['efficiency_divergence'] = data['range_efficiency'] - data['volume_efficiency']
    
    # Adaptive signal combination
    data['range_vol_signal'] = data['efficiency_divergence'] * data['range_percentile_10d']
    
    # Final alpha factor combination
    # Normalize components
    data['pv_signal_norm'] = data['pv_signal'] / (data['pv_signal'].rolling(window=20, min_periods=10).std() + 1e-8)
    data['breakout_signal_norm'] = data['breakout_signal'] / (abs(data['breakout_signal']).rolling(window=20, min_periods=10).std() + 1e-8)
    data['eff_mom_signal_norm'] = data['eff_mom_signal'] / (data['eff_mom_signal'].rolling(window=20, min_periods=10).std() + 1e-8)
    data['range_vol_signal_norm'] = data['range_vol_signal'] / (data['range_vol_signal'].rolling(window=20, min_periods=10).std() + 1e-8)
    
    # Volatility-adjusted weights
    vol_weights = {
        2: [0.4, 0.2, 0.3, 0.1],  # High volatility: emphasize divergence reversals
        1: [0.25, 0.25, 0.25, 0.25],  # Medium volatility: balanced
        0: [0.1, 0.4, 0.2, 0.3]   # Low volatility: emphasize breakouts
    }
    
    # Apply regime-specific weights
    alpha_values = []
    for idx, row in data.iterrows():
        regime = row['vol_regime']
        weights = vol_weights.get(regime, [0.25, 0.25, 0.25, 0.25])
        
        pv = row['pv_signal_norm'] if not pd.isna(row['pv_signal_norm']) else 0
        breakout = row['breakout_signal_norm'] if not pd.isna(row['breakout_signal_norm']) else 0
        eff_mom = row['eff_mom_signal_norm'] if not pd.isna(row['eff_mom_signal_norm']) else 0
        range_vol = row['range_vol_signal_norm'] if not pd.isna(row['range_vol_signal_norm']) else 0
        
        alpha = (weights[0] * pv + weights[1] * breakout + 
                weights[2] * eff_mom + weights[3] * range_vol)
        alpha_values.append(alpha)
    
    data['alpha'] = alpha_values
    
    # Clean up intermediate columns
    result = data['alpha'].copy()
    result.name = 'heuristics_v2_alpha'
    
    return result
