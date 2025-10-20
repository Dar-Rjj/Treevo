import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Price-Volume Asymmetry & Acceleration Dynamics alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Basic price and volume calculations
    df['returns'] = df['close'].pct_change()
    df['prev_close'] = df['close'].shift(1)
    df['day_range'] = df['high'] - df['low']
    df['range_mid'] = (df['high'] + df['low']) / 2
    
    # 1. Asymmetric Volume-Price Relationship Analysis
    df['is_up_day'] = (df['close'] > df['prev_close']).astype(int)
    df['is_down_day'] = (df['close'] < df['prev_close']).astype(int)
    
    # Rolling volume averages for up/down days
    df['up_day_volume_avg'] = df['volume'].rolling(window=20).apply(
        lambda x: x[df['is_up_day'].iloc[-len(x):] == 1].mean() if (df['is_up_day'].iloc[-len(x):] == 1).any() else np.nan, 
        raw=False
    )
    df['down_day_volume_avg'] = df['volume'].rolling(window=20).apply(
        lambda x: x[df['is_down_day'].iloc[-len(x):] == 1].mean() if (df['is_down_day'].iloc[-len(x):] == 1).any() else np.nan, 
        raw=False
    )
    
    # Volume asymmetry ratio
    df['volume_asymmetry'] = df['up_day_volume_avg'] / df['down_day_volume_avg']
    
    # Price-range volume efficiency
    df['volume_efficiency'] = df['volume'] / (df['day_range'] + 1e-8)
    
    # Volume concentration in upper/lower range
    df['in_upper_range'] = (df['close'] > df['range_mid']).astype(int)
    df['upper_range_volume_ratio'] = df['volume'].rolling(window=10).apply(
        lambda x: x[df['in_upper_range'].iloc[-len(x):] == 1].sum() / x.sum() if x.sum() > 0 else 0,
        raw=False
    )
    
    # Volume persistence asymmetry
    df['volume_change'] = df['volume'].pct_change()
    df['consecutive_up_volume_accel'] = df['volume_change'].rolling(window=5).apply(
        lambda x: x[(df['is_up_day'].iloc[-len(x):] == 1) & (df['is_up_day'].iloc[-len(x):].shift(1) == 1)].mean() if len(x) > 1 else np.nan,
        raw=False
    )
    
    # 2. Acceleration-Deceleration Momentum Framework
    # Price acceleration metrics
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['price_acceleration'] = df['momentum_5'] - df['momentum_10']
    
    # Volume acceleration patterns
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_momentum'] = (df['volume_ma_5'] - df['volume_ma_10']) / df['volume_ma_10']
    df['volume_acceleration'] = df['volume_momentum'].diff(3)
    
    # Cross-acceleration analysis
    df['acceleration_alignment'] = np.sign(df['price_acceleration']) * np.sign(df['volume_acceleration'])
    
    # 3. Regime-Dependent Signal Interpretation
    # Volatility regime classification
    df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
    df['atr_median_60'] = df['atr'].rolling(window=60).median()
    df['high_vol_regime'] = (df['atr'] > df['atr_median_60']).astype(int)
    
    # Volume regime identification
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['high_volume_regime'] = (df['volume'] > df['volume_ma_20']).astype(int)
    
    # Trend regime assessment
    df['momentum_short'] = df['close'].pct_change(5)
    df['momentum_medium'] = df['close'].pct_change(20)
    df['trend_alignment'] = np.sign(df['momentum_short']) + np.sign(df['momentum_medium'])
    df['strong_trend'] = (abs(df['trend_alignment']) == 2).astype(int)
    
    # 4. Asymmetry-Based Predictive Patterns & Adaptive Alpha Construction
    # Core asymmetry factors
    df['volume_asymmetry_score'] = df['volume_asymmetry'].rolling(window=10).mean()
    df['price_accel_strength'] = df['price_acceleration'].abs().rolling(window=5).mean()
    df['volume_accel_momentum'] = df['volume_acceleration'].rolling(window=5).mean()
    
    # Regime-conditional weighting
    # Low volatility: emphasize asymmetry signals
    low_vol_weight = (1 - df['high_vol_regime']) * 0.6
    # High volatility: emphasize acceleration patterns  
    high_vol_weight = df['high_vol_regime'] * 0.7
    # Volume regime adjustment
    volume_regime_weight = df['high_volume_regime'] * 0.4 + (1 - df['high_volume_regime']) * 0.3
    # Trend context integration
    trend_weight = df['strong_trend'] * 0.5 + (1 - df['strong_trend']) * 0.3
    
    # Composite alpha construction
    # Base asymmetry component
    asymmetry_component = (
        df['volume_asymmetry_score'].fillna(0) * 0.3 +
        df['upper_range_volume_ratio'].fillna(0) * 0.2 +
        df['consecutive_up_volume_accel'].fillna(0) * 0.2
    )
    
    # Acceleration component
    acceleration_component = (
        df['price_accel_strength'].fillna(0) * 0.4 +
        df['volume_accel_momentum'].fillna(0) * 0.3 +
        df['acceleration_alignment'].fillna(0) * 0.3
    )
    
    # Regime-adaptive combination
    regime_weight = (low_vol_weight + high_vol_weight + volume_regime_weight + trend_weight) / 4
    
    # Final composite alpha
    alpha = (
        asymmetry_component * (1 - regime_weight) +
        acceleration_component * regime_weight
    )
    
    # Normalize and handle edge cases
    alpha = (alpha - alpha.rolling(window=60).mean()) / (alpha.rolling(window=60).std() + 1e-8)
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha
