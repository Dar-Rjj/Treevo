import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate composite alpha factor using multiple technical approaches
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Momentum-Volume Divergence Factor
    # Price momentum calculations
    data['mom_10'] = data['close'] / data['close'].shift(10) - 1
    data['mom_5'] = data['close'] / data['close'].shift(5) - 1
    
    # Volume momentum calculations
    data['vol_mom_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['vol_mom_5'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Momentum divergence
    data['price_mom_trend'] = np.sign(data['mom_5'] - data['mom_10'])
    data['vol_mom_trend'] = np.sign(data['vol_mom_5'] - data['vol_mom_10'])
    data['momentum_divergence'] = data['price_mom_trend'] * data['vol_mom_trend']
    
    # 2. High-Low Range Efficiency Factor
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Price movement efficiency
    data['c2c_return'] = abs(data['close'] / data['close'].shift(1) - 1)
    data['efficiency'] = data['c2c_return'] / data['true_range']
    data['efficiency_8d'] = data['efficiency'].rolling(window=8).mean()
    data['efficiency_3d'] = data['efficiency'].rolling(window=3).mean()
    data['range_efficiency'] = data['efficiency_3d'] / data['efficiency_8d']
    
    # 3. Volume-Price Correlation Regime Factor
    # Returns and volume changes
    data['returns'] = data['close'].pct_change()
    data['vol_change'] = data['volume'].pct_change()
    
    # Rolling correlation
    data['corr_6d'] = data['returns'].rolling(window=6).corr(data['vol_change'])
    
    # Correlation regime identification
    data['strong_pos_corr'] = (data['corr_6d'] > 0.3).astype(int)
    data['strong_neg_corr'] = (data['corr_6d'] < -0.3).astype(int)
    
    # Regime persistence
    data['corr_regime'] = np.where(data['corr_6d'] > 0.3, 1, 
                                  np.where(data['corr_6d'] < -0.3, -1, 0))
    data['regime_persistence'] = data['corr_regime'].rolling(window=5).sum()
    data['correlation_regime'] = data['corr_6d'] * data['regime_persistence']
    
    # 4. Close Position Relative to Daily Range Factor
    # Daily range and close position
    data['daily_range'] = data['high'] - data['low']
    data['close_position'] = (data['close'] - data['low']) / data['daily_range']
    
    # Position analysis
    data['avg_position_5d'] = data['close_position'].rolling(window=5).mean()
    data['position_deviation'] = data['close_position'] - data['avg_position_5d']
    data['position_extreme'] = abs(data['position_deviation'])
    
    # Mean-reversion signal (negative for extremes)
    data['position_signal'] = -data['position_extreme'] * np.sign(data['position_deviation'])
    
    # 5. Volume-Weighted Price Acceleration Factor
    # Price acceleration (second derivative)
    data['price_velocity'] = data['close'].pct_change()
    data['price_acceleration'] = data['price_velocity'].diff()
    
    # Volume acceleration
    data['volume_velocity'] = data['volume'].pct_change()
    data['volume_acceleration'] = data['volume_velocity'].diff()
    
    # Composite signal
    data['acceleration_magnitude'] = abs(data['price_acceleration'])
    data['volume_weighted_accel'] = data['price_acceleration'] * data['volume_acceleration']
    data['vw_accel_signal'] = data['volume_weighted_accel'] * data['acceleration_magnitude']
    
    # 6. Intraday Volatility Persistence Factor
    # Intraday volatility
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    # Volatility persistence (autocorrelation)
    data['vol_lag1'] = data['intraday_vol'].shift(1)
    data['vol_lag2'] = data['intraday_vol'].shift(2)
    vol_rolling = data['intraday_vol'].rolling(window=7)
    data['vol_persistence'] = vol_rolling.apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Persistence signal
    data['vol_persistence_signal'] = data['vol_persistence'] * data['intraday_vol']
    
    # 7. Amount-Based Order Flow Imbalance Factor
    # Order flow calculations
    data['amount_per_share'] = data['amount'] / data['volume']
    data['order_flow_ratio'] = data['amount_per_share'] / data['amount_per_share'].rolling(window=10).mean()
    
    # Order flow imbalance
    data['order_imbalance'] = data['order_flow_ratio'] - 1
    data['imbalance_persistence'] = data['order_imbalance'].rolling(window=5).sum()
    data['order_flow_signal'] = data['order_imbalance'] * data['imbalance_persistence']
    
    # 8. Multi-Timeframe Momentum Convergence Factor
    # Multi-timeframe momentum
    data['mom_3'] = data['close'] / data['close'].shift(3) - 1
    data['mom_15'] = data['close'] / data['close'].shift(15) - 1
    
    # Momentum convergence
    short_term_avg = (data['mom_3'] + data['mom_5']) / 2
    medium_term_avg = (data['mom_10'] + data['mom_15']) / 2
    data['momentum_alignment'] = np.sign(short_term_avg) * np.sign(medium_term_avg)
    data['convergence_strength'] = abs(short_term_avg - medium_term_avg)
    data['multi_timeframe_signal'] = data['momentum_alignment'] * data['convergence_strength']
    
    # Combine all factors with equal weights
    factors = [
        'momentum_divergence', 'range_efficiency', 'correlation_regime',
        'position_signal', 'vw_accel_signal', 'vol_persistence_signal',
        'order_flow_signal', 'multi_timeframe_signal'
    ]
    
    # Z-score normalization for each factor
    for factor in factors:
        if factor in data.columns:
            mean_val = data[factor].rolling(window=20).mean()
            std_val = data[factor].rolling(window=20).std()
            data[f'{factor}_norm'] = (data[factor] - mean_val) / std_val
    
    # Composite factor (equal weighted average of normalized factors)
    normalized_factors = [f'{factor}_norm' for factor in factors if f'{factor}_norm' in data.columns]
    composite_factor = data[normalized_factors].mean(axis=1)
    
    return composite_factor
