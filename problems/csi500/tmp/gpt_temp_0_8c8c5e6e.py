import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Volume-Weighted Reversal factor
    Combines volatility-normalized reversal signals with volume divergence analysis
    and adaptive regime detection for improved return prediction
    """
    df = data.copy()
    
    # Multi-Timeframe Returns
    df['return_1d'] = df['close'] / df['close'].shift(1) - 1
    df['return_3d'] = df['close'] / df['close'].shift(3) - 1
    df['return_5d'] = df['close'] / df['close'].shift(5) - 1
    
    # Volatility Estimation
    df['vol_10d'] = df['return_1d'].rolling(window=10, min_periods=5).std()
    df['vol_20d'] = df['return_1d'].rolling(window=20, min_periods=10).std()
    df['vol_5d'] = df['return_1d'].rolling(window=5, min_periods=3).std()
    
    # Volatility-Normalized Reversal
    df['reversal_short'] = -1 * (df['return_3d'] / df['vol_10d'])
    df['reversal_medium'] = -1 * (df['return_5d'] / df['vol_20d'])
    
    # Volume Acceleration Signals
    df['volume_momentum_3d'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    
    # Price-Volume Divergence
    df['volume_acceleration_ratio'] = df['volume_momentum_3d'] / df['volume_momentum_10d']
    df['price_volume_divergence'] = df['return_5d'] * df['volume_acceleration_ratio']
    
    # Nonlinear Volume Weighting
    df['volume_rank'] = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['exp_volume_scaling'] = np.exp(df['volume_rank'] / 100)
    df['volume_weight'] = df['exp_volume_scaling'] / df['exp_volume_scaling'].rolling(window=20, min_periods=10).mean()
    
    # Regime Detection & Adaptation
    df['volatility_ratio'] = df['vol_5d'] / df['vol_20d']
    df['regime_strength'] = np.abs(df['volatility_ratio'] - 1)
    
    # Market State Indicators
    df['volume_sma_50'] = df['volume'].rolling(window=50, min_periods=25).mean()
    df['volume_vs_50d'] = df['volume'] / df['volume_sma_50'].shift(1) - 1
    
    # 5-day High-Low Range
    df['high_5d'] = df['high'].rolling(window=5, min_periods=3).max()
    df['low_5d'] = df['low'].rolling(window=5, min_periods=3).min()
    df['price_range_5d'] = (df['high_5d'] - df['low_5d']) / df['close'].shift(5)
    
    # Base Reversal Selection based on Regime
    df['base_reversal'] = np.where(
        df['volatility_ratio'] > 1,  # High Volatility Regime
        df['reversal_short'],        # Use short-term reversal
        df['reversal_medium']        # Use medium-term reversal
    )
    
    # Volume Divergence Enhancement
    df['enhanced_reversal'] = df['base_reversal'] * df['price_volume_divergence'] * df['volume_weight']
    
    # Regime Strength Modulation
    df['final_alpha'] = df['enhanced_reversal'] * df['regime_strength']
    
    # Clean up and return
    result = df['final_alpha'].replace([np.inf, -np.inf], np.nan)
    return result
