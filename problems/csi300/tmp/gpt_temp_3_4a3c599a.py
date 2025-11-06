import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate mid-price
    df['mid_price'] = (df['high'] + df['low']) / 2
    
    # Volatility Regime Analysis
    # Bidirectional Volatility
    returns = df['mid_price'].pct_change()
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    upside_vol = positive_returns.rolling(window=30, min_periods=15).std()
    downside_vol = negative_returns.rolling(window=30, min_periods=15).std()
    
    # Multi-Timeframe Range Volatility
    daily_range = (df['high'] - df['low']) / df['close']
    short_term_vol = daily_range.rolling(window=5, min_periods=3).mean()
    medium_term_vol = daily_range.rolling(window=10, min_periods=5).mean()
    long_term_vol = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Multi-Scale Acceleration Dynamics
    # Price Acceleration Components
    # Short-Term Price Acceleration
    price_5d_return_t = (df['mid_price'] - df['mid_price'].shift(5)) / df['mid_price'].shift(5)
    price_5d_return_t3 = (df['mid_price'].shift(3) - df['mid_price'].shift(8)) / df['mid_price'].shift(8)
    short_price_accel = price_5d_return_t - price_5d_return_t3
    
    # Medium-Term Price Acceleration
    price_10d_return_t = (df['mid_price'] - df['mid_price'].shift(10)) / df['mid_price'].shift(10)
    price_10d_return_t5 = (df['mid_price'].shift(5) - df['mid_price'].shift(15)) / df['mid_price'].shift(15)
    medium_price_accel = price_10d_return_t - price_10d_return_t5
    
    # Volume Acceleration Components
    # Short-Term Volume Acceleration
    vol_5d_return_t = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5).replace(0, np.nan)
    vol_5d_return_t3 = (df['volume'].shift(3) - df['volume'].shift(8)) / df['volume'].shift(8).replace(0, np.nan)
    short_vol_accel = vol_5d_return_t - vol_5d_return_t3
    
    # Medium-Term Volume Acceleration
    vol_10d_return_t = (df['volume'] - df['volume'].shift(10)) / df['volume'].shift(10).replace(0, np.nan)
    vol_10d_return_t5 = (df['volume'].shift(5) - df['volume'].shift(15)) / df['volume'].shift(15).replace(0, np.nan)
    medium_vol_accel = vol_10d_return_t - vol_10d_return_t5
    
    # Efficiency Divergence Calculation
    # Price momentum efficiency
    price_momentum_3d = df['mid_price'].pct_change(3) / returns.rolling(window=3, min_periods=2).std()
    price_momentum_8d = df['mid_price'].pct_change(8) / returns.rolling(window=8, min_periods=4).std()
    price_momentum_20d = df['mid_price'].pct_change(20) / returns.rolling(window=20, min_periods=10).std()
    
    # Volume flow efficiency
    vol_flow_3d = ((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan) * df['volume']).rolling(window=3, min_periods=2).mean()
    vol_flow_8d = ((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan) * df['volume']).rolling(window=8, min_periods=4).mean()
    vol_flow_20d = ((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan) * df['volume']).rolling(window=20, min_periods=10).mean()
    
    # Acceleration-Efficiency Divergence
    short_term_div = np.sign(short_price_accel - short_vol_accel) * vol_flow_3d
    medium_term_div = np.sign(medium_price_accel - medium_vol_accel) * price_momentum_8d
    
    # Cross-Timeframe Pressure Analysis
    short_term_pressure = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    medium_term_pressure = short_term_pressure.rolling(window=5, min_periods=3).mean() / short_term_pressure.rolling(window=10, min_periods=5).mean()
    
    # Pressure Consistency Analysis
    pressure_signs = pd.concat([
        short_term_pressure.rolling(window=3).apply(lambda x: np.sign(x).mean()),
        medium_term_pressure.rolling(window=3).apply(lambda x: np.sign(x).mean())
    ], axis=1).mean(axis=1)
    
    # Regime-Adaptive Weighting
    vol_asymmetry = upside_vol / downside_vol.replace(0, np.nan)
    current_vol = daily_range.rolling(window=5).mean()
    historical_vol = daily_range.rolling(window=60).mean()
    vol_level = current_vol / historical_vol.replace(0, np.nan)
    
    # Volatility regime classification
    vol_regime = pd.cut(vol_level, bins=[0, 0.7, 1.3, float('inf')], labels=['low', 'normal', 'high'])
    
    # Regime-Weighted Divergence
    regime_weighted_div = pd.Series(index=df.index, dtype=float)
    
    high_vol_mask = vol_regime == 'high'
    low_vol_mask = vol_regime == 'low'
    normal_vol_mask = vol_regime == 'normal'
    
    regime_weighted_div[high_vol_mask] = (short_term_div * 0.6 + medium_term_div * 0.3 + pressure_signs * 0.1)[high_vol_mask]
    regime_weighted_div[low_vol_mask] = (short_term_div * 0.2 + medium_term_div * 0.3 + pressure_signs * 0.5)[low_vol_mask]
    regime_weighted_div[normal_vol_mask] = (short_term_div * 0.33 + medium_term_div * 0.33 + pressure_signs * 0.34)[normal_vol_mask]
    
    # Intraday Efficiency Enhancement
    efficiency_base = (df['close'] - df['open']) / pd.concat([
        (df['high'] - df['low']),
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1).replace(0, np.nan)
    
    volatility_regime_strength = (upside_vol - downside_vol).abs() / (upside_vol + downside_vol).replace(0, np.nan)
    regime_adjusted_efficiency = efficiency_base * (2 - volatility_regime_strength)
    
    # Volume Efficiency
    volume_efficiency = df['amount'] / (df['volume'] * df['close']).replace(0, np.nan)
    
    # Composite Alpha Generation
    base_divergence_signal = regime_weighted_div * np.sign(df['close'] - df['close'].shift(5))
    efficiency_multiplier = regime_adjusted_efficiency * volume_efficiency.rolling(window=5).mean()
    
    # Volume Confidence
    volume_ratio = df['volume'] / df['volume'].rolling(window=20).mean()
    volume_momentum = df['volume'].pct_change(5)
    volume_confidence = volume_ratio * volume_momentum
    
    # Final Alpha Factor
    final_alpha = base_divergence_signal * efficiency_multiplier * volume_confidence
    
    return final_alpha
