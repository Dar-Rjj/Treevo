import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Temporal Information Asymmetry Factor combining intraday patterns, volume-time distribution,
    price discovery efficiency, and multi-timeframe information processing.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic intraday metrics
    df['daily_range'] = df['high'] - df['low']
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # 1. Intraday Information Flow Patterns
    # Early momentum persistence proxy (using first hour approximation)
    df['early_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['early_momentum_smooth'] = df['early_momentum'].rolling(window=5, min_periods=3).mean()
    
    # Late session reversal strength (volume-price divergence)
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['volume_price_divergence'] = df['price_change'] * df['volume_change'].shift(1)
    df['late_reversal_strength'] = df['volume_price_divergence'].rolling(window=3).sum()
    
    # 2. Volume-Time Distribution Analysis
    # Volume concentration asymmetry (using rolling windows to simulate time segments)
    morning_volume_ratio = df['volume'].rolling(window=5).apply(
        lambda x: x[:3].sum() / (x.sum() + 1e-8) if len(x) == 5 else np.nan
    )
    afternoon_volume_ratio = df['volume'].rolling(window=5).apply(
        lambda x: x[3:].sum() / (x.sum() + 1e-8) if len(x) == 5 else np.nan
    )
    df['volume_concentration_asymmetry'] = morning_volume_ratio - afternoon_volume_ratio
    
    # Liquidity evaporation timing (consecutive low volume periods)
    volume_ma = df['volume'].rolling(window=10).mean()
    low_volume_flag = (df['volume'] < volume_ma * 0.7).astype(int)
    df['liquidity_evaporation'] = low_volume_flag.rolling(window=3).sum()
    
    # 3. Price Discovery Efficiency
    # Opening price efficiency (opening vs VWAP deviation)
    df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan)
    df['opening_efficiency'] = (df['open'] - df['vwap']) / df['vwap']
    df['opening_efficiency_smooth'] = df['opening_efficiency'].rolling(window=5).mean()
    
    # Intraday price continuity (volatility per volume unit)
    df['price_continuity'] = (df['high'] - df['low']) / (df['volume'] + 1e-8)
    df['price_continuity_norm'] = df['price_continuity'] / df['price_continuity'].rolling(window=20).mean()
    
    # 4. Multi-Timeframe Information Processing
    # Short-term memory effects (same-hour correlations proxy)
    df['intraday_momentum'] = df['intraday_return'].rolling(window=3).mean()
    df['delayed_reaction'] = (df['volume'].shift(1) / df['volume'].shift(2)) * df['price_change'].shift(1)
    
    # Cross-temporal interference (overnight gaps vs intraday trend)
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_trend = (df['close'] - df['open']) / df['open']
    df['cross_temporal_interference'] = overnight_gap * intraday_trend
    
    # Combine components with appropriate weights
    factor = (
        0.25 * df['early_momentum_smooth'] +
        0.20 * df['late_reversal_strength'] +
        0.15 * df['volume_concentration_asymmetry'] +
        0.10 * df['liquidity_evaporation'] +
        0.15 * df['opening_efficiency_smooth'] +
        0.10 * df['price_continuity_norm'] +
        0.05 * df['intraday_momentum'] +
        0.05 * df['cross_temporal_interference']
    )
    
    # Normalize the factor
    factor_ma = factor.rolling(window=20).mean()
    factor_std = factor.rolling(window=20).std()
    normalized_factor = (factor - factor_ma) / (factor_std + 1e-8)
    
    return normalized_factor
