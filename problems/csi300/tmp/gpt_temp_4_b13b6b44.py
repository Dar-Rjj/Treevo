import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Microstructure-Regime Adaptive Momentum Asymmetry Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price returns and accelerations
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10).shift(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['ret_40d'] = df['close'].pct_change(40).shift(20)
    
    # Multi-scale momentum acceleration
    df['accel_5d'] = df['ret_5d'] - df['ret_10d']
    df['accel_20d'] = df['ret_20d'] - df['ret_40d']
    
    # Calculate ATR for volatility regimes
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_5'] = df['tr'].rolling(window=5).mean()
    df['atr_20'] = df['tr'].rolling(window=20).mean()
    
    # Volatility regime classification
    df['vol_regime'] = np.where(
        df['atr_5'] / df['atr_20'] > 1.2, 'high',
        np.where(df['atr_5'] / df['atr_20'] < 0.8, 'low', 'transition')
    )
    
    # Acceleration asymmetry patterns
    df['up_accel_strength'] = np.maximum(0, df['accel_5d']) * (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['down_accel_strength'] = np.maximum(0, -df['accel_5d']) * (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['accel_asymmetry_ratio'] = df['up_accel_strength'] / (df['down_accel_strength'] + 1e-8)
    
    # Volatility-adjusted acceleration
    df['vol_adjusted_accel'] = np.where(
        df['vol_regime'] == 'high',
        df['accel_5d'] * (df['atr_5'] / df['atr_20']),
        np.where(
            df['vol_regime'] == 'low',
            df['accel_5d'] * (df['atr_20'] / df['atr_5']),
            df['accel_5d'] * (1 + abs(df['atr_5'] / df['atr_20'] - 1))
        )
    )
    
    # Amount-flow momentum components
    df['amount_ma_5'] = df['amount'].rolling(window=5).mean()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    
    # Directional amount pressure
    df['bullish_amount_momentum'] = (df['close'] - df['open']) * (df['amount'] - df['amount_ma_5']) / (df['amount_ma_5'] + 1e-8)
    df['bearish_amount_momentum'] = (df['open'] - df['close']) * (df['amount'] - df['amount_ma_5']) / (df['amount_ma_5'] + 1e-8)
    df['net_amount_pressure'] = df['bullish_amount_momentum'] - df['bearish_amount_momentum']
    
    # Trade size patterns
    df['trade_size'] = df['amount'] / (df['volume'] + 1e-8)
    df['trade_size_ma_5'] = df['trade_size'].rolling(window=5).mean()
    df['trade_size_momentum'] = df['trade_size'] / df['trade_size'].shift(1) - 1
    
    # Flow concentration momentum
    df['amount_std_5'] = df['amount'].rolling(window=5).std()
    df['volume_std_5'] = df['volume'].rolling(window=5).std()
    df['amount_dist_momentum'] = (df['amount'] - df['amount_ma_5']) / (df['amount_std_5'] + 1e-8)
    df['volume_dist_momentum'] = (df['volume'] - df['volume_ma_5']) / (df['volume_std_5'] + 1e-8)
    df['flow_momentum_divergence'] = df['amount_dist_momentum'] - df['volume_dist_momentum']
    
    # Flow regime classification
    df['flow_regime'] = np.where(
        (df['trade_size_momentum'] > 0) & (df['trade_size'] > df['trade_size'].rolling(window=10).mean()),
        'institutional',
        np.where(
            (df['volume'] > df['volume'].rolling(window=10).mean()) & (df['trade_size_momentum'] < 0),
            'retail',
            'balanced'
        )
    )
    
    # Microstructure fracture detection
    df['accel_change'] = abs(df['accel_5d'] - df['accel_5d'].shift(1))
    df['range_change'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) - 1
    df['volume_spike'] = df['volume'] / df['volume_ma_5'] - 1
    df['trade_size_anomaly'] = df['trade_size'] / df['trade_size_ma_5'] - 1
    
    # Multi-timeframe convergence
    df['accel_direction_agreement'] = (np.sign(df['accel_5d']) == np.sign(df['accel_20d'])).astype(int)
    
    # Regime-adaptive weighting
    for idx in df.index:
        if pd.isna(df.loc[idx, 'accel_5d']) or pd.isna(df.loc[idx, 'accel_20d']):
            result.loc[idx] = 0
            continue
            
        # Core components
        price_asymmetry = df.loc[idx, 'accel_asymmetry_ratio']
        vol_adjusted_momentum = df.loc[idx, 'vol_adjusted_accel']
        amount_pressure = df.loc[idx, 'net_amount_pressure']
        flow_divergence = df.loc[idx, 'flow_momentum_divergence']
        
        # Regime-specific weights
        vol_regime = df.loc[idx, 'vol_regime']
        flow_regime = df.loc[idx, 'flow_regime']
        
        # Volatility regime weights
        if vol_regime == 'high':
            vol_weight = 1.2
            fracture_weight = 1.5
        elif vol_regime == 'low':
            vol_weight = 0.8
            fracture_weight = 0.5
        else:
            vol_weight = 1.0
            fracture_weight = 1.0
        
        # Flow regime weights
        if flow_regime == 'institutional':
            flow_weight = 1.3
            amount_weight = 1.5
        elif flow_regime == 'retail':
            flow_weight = 0.7
            amount_weight = 0.5
        else:
            flow_weight = 1.0
            amount_weight = 1.0
        
        # Quality metrics
        direction_agreement = df.loc[idx, 'accel_direction_agreement']
        regime_stability = 1.0  # Simplified stability measure
        
        # Composite factor calculation
        factor_value = (
            price_asymmetry * vol_weight * 0.3 +
            vol_adjusted_momentum * vol_weight * 0.25 +
            amount_pressure * amount_weight * 0.2 +
            flow_divergence * flow_weight * 0.15 +
            direction_agreement * regime_stability * 0.1
        )
        
        # Apply quality filter
        if direction_agreement == 0:
            factor_value *= 0.5
        
        result.loc[idx] = factor_value
    
    # Normalize the final factor
    result = (result - result.rolling(window=20).mean()) / (result.rolling(window=20).std() + 1e-8)
    
    return result
