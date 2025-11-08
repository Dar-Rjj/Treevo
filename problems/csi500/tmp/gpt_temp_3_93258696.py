import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns and basic price features
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Volatility regime calculation
    df['volatility_10d'] = df['returns'].rolling(window=10).std()
    df['range_volatility_10d'] = df['high_low_range'].rolling(window=10).std()
    df['medium_term_vol'] = df['volatility_10d'].rolling(window=30).mean()
    
    # Volatility regime classification
    conditions = [
        df['volatility_10d'] > 1.5 * df['medium_term_vol'],
        df['volatility_10d'] < 0.7 * df['medium_term_vol']
    ]
    choices = [2, 0]  # 2: high, 0: low, 1: normal
    df['vol_regime'] = np.select(conditions, choices, default=1)
    
    # Regime persistence
    df['regime_persistence'] = 0
    for i in range(1, len(df)):
        if df['vol_regime'].iloc[i] == df['vol_regime'].iloc[i-1]:
            df['regime_persistence'] = df['regime_persistence'].iloc[i-1] + 1
    
    # Regime-specific momentum calculations
    # High volatility momentum
    df['gap_capture_momentum'] = np.where(
        df['overnight_gap'] * df['intraday_return'] > 0,
        df['intraday_return'] / (1 + abs(df['overnight_gap'])),
        -df['intraday_return'] / (1 + abs(df['overnight_gap']))
    )
    df['intraday_reversal'] = -df['returns'].rolling(window=3).mean()
    df['vol_adj_return'] = df['returns'] / (df['volatility_10d'] + 1e-8)
    
    # Low volatility momentum
    df['range_expansion'] = (df['high_low_range'] - df['high_low_range'].rolling(window=5).mean()) / df['high_low_range'].rolling(window=5).std()
    df['volume_breakout'] = (df['volume'] - df['volume'].rolling(window=10).mean()) / df['volume'].rolling(window=10).std()
    df['micro_trend'] = df['close'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / (x.std() + 1e-8))
    
    # Normal volatility momentum
    df['price_momentum'] = df['close'].pct_change(periods=5)
    df['volume_weighted_return'] = (df['returns'] * df['volume']).rolling(window=5).mean()
    df['range_efficiency'] = abs(df['returns']) / (df['high_low_range'] + 1e-8)
    
    # Adaptive momentum blending
    high_vol_momentum = 0.4 * df['gap_capture_momentum'] + 0.4 * df['intraday_reversal'] + 0.2 * df['vol_adj_return']
    low_vol_momentum = 0.5 * df['range_expansion'] + 0.3 * df['volume_breakout'] + 0.2 * df['micro_trend']
    normal_vol_momentum = 0.5 * df['price_momentum'] + 0.3 * df['volume_weighted_return'] + 0.2 * df['range_efficiency']
    
    # Regime confidence weights
    regime_confidence = np.minimum(df['regime_persistence'] / 5, 1.0)
    
    df['regime_momentum'] = np.where(
        df['vol_regime'] == 2,
        regime_confidence * high_vol_momentum + (1 - regime_confidence) * normal_vol_momentum,
        np.where(
            df['vol_regime'] == 0,
            regime_confidence * low_vol_momentum + (1 - regime_confidence) * normal_vol_momentum,
            normal_vol_momentum
        )
    )
    
    # Microstructure liquidity factor
    # Spread proxy construction
    df['effective_spread'] = df['high_low_range'] / (abs(df['returns']) + 1e-8)
    df['gap_range_ratio'] = abs(df['overnight_gap']) / (df['high_low_range'] + 1e-8)
    
    # Spread dynamics
    df['spread_change'] = df['effective_spread'].pct_change()
    df['spread_volatility'] = df['effective_spread'].rolling(window=10).std()
    df['price_volatility'] = df['returns'].rolling(window=10).std()
    df['spread_vol_ratio'] = df['spread_volatility'] / (df['price_volatility'] + 1e-8)
    
    # Liquidity regimes
    spread_conditions = [
        (df['effective_spread'] < df['effective_spread'].rolling(window=20).quantile(0.3)) & 
        (df['spread_volatility'] < df['spread_volatility'].rolling(window=20).quantile(0.3)),
        (df['effective_spread'] > df['effective_spread'].rolling(window=20).quantile(0.7)) | 
        (df['spread_volatility'] > df['spread_volatility'].rolling(window=20).quantile(0.7))
    ]
    spread_choices = [2, 0]  # 2: high liquidity, 0: low liquidity, 1: normal
    df['liquidity_regime'] = np.select(spread_conditions, spread_choices, default=1)
    
    # Volume-liquidity interaction
    df['volume_concentration'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['liquidity_volume_momentum'] = df['volume_concentration'] * (1 - df['spread_change'])
    df['price_impact'] = abs(df['returns']) / (df['amount'] + 1e-8)
    
    # Liquidity imbalance signal
    df['spread_volume_divergence'] = np.where(
        (df['spread_change'] < 0) & (df['volume_concentration'] > 1),
        df['volume_concentration'] * (1 - df['spread_change']),
        np.where(
            (df['spread_change'] > 0) & (df['volume_concentration'] < 1),
            -df['volume_concentration'] * (1 + df['spread_change']),
            0
        )
    )
    
    df['liquidity_momentum_alignment'] = df['regime_momentum'] * (1 - df['spread_change'])
    
    # Composite liquidity factor
    df['liquidity_factor'] = (
        0.6 * df['spread_volume_divergence'] + 
        0.4 * df['liquidity_momentum_alignment']
    )
    
    # Adaptive microstructure composite
    # Volatility-regime weighted components
    high_vol_weight = np.where(df['vol_regime'] == 2, 0.7, 0.3)
    low_vol_weight = np.where(df['vol_regime'] == 0, 0.7, 0.3)
    normal_vol_weight = np.where(df['vol_regime'] == 1, 0.7, 0.3)
    
    # Final adaptive alpha factor
    df['adaptive_alpha'] = (
        high_vol_weight * df['regime_momentum'] * df['liquidity_factor'] +
        low_vol_weight * df['regime_momentum'] * df['liquidity_factor'] +
        normal_vol_weight * df['regime_momentum'] * df['liquidity_factor']
    ) / 3
    
    return df['adaptive_alpha']
