import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum Divergence Framework alpha factor
    Combines hierarchical momentum, volume asymmetry, and intraday structure
    """
    # Make copy to avoid modifying original dataframe
    data = df.copy()
    
    # FRACTAL MOMENTUM STRUCTURE
    # Hierarchical Price Momentum (using available data frequencies)
    # Micro-scale: 5-day rolling returns (proxy for intraday)
    data['micro_momentum'] = data['close'].pct_change(periods=5)
    
    # Meso-scale: 10-day rolling returns
    data['meso_momentum'] = data['close'].pct_change(periods=10)
    
    # Macro-scale: 20-day rolling returns
    data['macro_momentum'] = data['close'].pct_change(periods=20)
    
    # Momentum Divergence Detection
    data['momentum_spread'] = data['micro_momentum'] - data['macro_momentum']
    data['momentum_acceleration'] = data['meso_momentum'].diff()
    
    # Multi-scale Momentum Persistence
    # Count consecutive same-direction momentum days for micro scale
    data['micro_momentum_sign'] = np.sign(data['micro_momentum'])
    data['micro_momentum_persistence'] = (
        data['micro_momentum_sign']
        .groupby((data['micro_momentum_sign'] != data['micro_momentum_sign'].shift()).cumsum())
        .cumcount() + 1
    ) * data['micro_momentum_sign']
    
    # Momentum consistency across scales
    data['momentum_consistency'] = (
        (np.sign(data['micro_momentum']) == np.sign(data['meso_momentum'])).astype(int) +
        (np.sign(data['meso_momentum']) == np.sign(data['macro_momentum'])).astype(int)
    )
    
    # VOLUME FLOW ASYMMETRY ANALYSIS
    # Bidirectional Volume Pressure
    # Up-day volume intensity
    data['up_day'] = data['close'] > data['open']
    data['up_volume_intensity'] = np.where(
        data['up_day'], 
        data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean(), 
        0
    )
    
    # Down-day volume intensity
    data['down_day'] = data['close'] < data['open']
    data['down_volume_intensity'] = np.where(
        data['down_day'], 
        data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean(), 
        0
    )
    
    # Large-tick volume concentration (using high-low range as proxy)
    data['tick_size'] = (data['high'] - data['low']) / data['close']
    data['large_tick_up'] = np.where(
        (data['up_day']) & (data['tick_size'] > data['tick_size'].rolling(window=20).median()),
        data['volume'], 
        0
    )
    data['large_tick_down'] = np.where(
        (data['down_day']) & (data['tick_size'] > data['tick_size'].rolling(window=20).median()),
        data['volume'], 
        0
    )
    
    # Volume Imbalance Dynamics
    data['net_volume_pressure'] = (
        data['large_tick_up'].rolling(window=5).sum() - 
        data['large_tick_down'].rolling(window=5).sum()
    ) / data['volume'].rolling(window=5).sum()
    
    # Volume pressure persistence
    data['volume_pressure_sign'] = np.sign(data['net_volume_pressure'])
    data['volume_pressure_persistence'] = (
        data['volume_pressure_sign']
        .groupby((data['volume_pressure_sign'] != data['volume_pressure_sign'].shift()).cumsum())
        .cumcount() + 1
    ) * data['volume_pressure_sign']
    
    # Volume-Momentum Divergence
    data['volume_momentum_divergence'] = (
        np.sign(data['net_volume_pressure']) != np.sign(data['micro_momentum'])
    ).astype(int)
    
    # INTRADAY STRUCTURE DECOMPOSITION
    # Opening Auction Dynamics
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['opening_range'] = (data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()) / data['close'].shift(1)
    
    # Midday Momentum Structure
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Intraday trend consistency
    data['intraday_trend'] = np.where(
        data['close'] > data['open'], 
        (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan),
        (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    )
    
    # Closing Auction Impact
    data['closing_volume_ratio'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['final_hour_momentum'] = (data['close'] - data['open']) / data['open']
    
    # MULTI-TIMEFRAME REGIME IDENTIFICATION
    # Volatility Clustering Patterns
    data['volatility'] = data['close'].pct_change().rolling(window=20).std()
    data['volatility_regime'] = data['volatility'] > data['volatility'].rolling(window=50).median()
    
    # Trend Regime Classification
    data['trend_strength'] = data['close'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )
    data['strong_trend'] = abs(data['trend_strength']) > 0.3
    
    # Liquidity Regime Assessment
    data['liquidity'] = data['volume'] * data['close']
    data['liquidity_regime'] = data['liquidity'] < data['liquidity'].rolling(window=50).quantile(0.3)
    
    # ADAPTIVE SIGNAL CONSTRUCTION
    # Multi-scale Momentum Integration
    momentum_signal = (
        0.4 * data['micro_momentum'].fillna(0) +
        0.3 * data['meso_momentum'].fillna(0) + 
        0.3 * data['macro_momentum'].fillna(0)
    ) * (1 + 0.1 * data['momentum_consistency'])
    
    # Volume-Price Convergence Scoring
    volume_signal = data['net_volume_pressure'].fillna(0) * (
        1 + 0.2 * data['volume_pressure_persistence'].fillna(0)
    )
    
    # Volume-momentum alignment score
    alignment_score = np.where(
        data['volume_momentum_divergence'] == 0,  # No divergence
        1.2,  # Boost when aligned
        0.8   # Penalty when diverging
    )
    
    # Intraday Structure Enhancement
    intraday_signal = (
        0.3 * data['opening_gap'].fillna(0) +
        0.4 * data['intraday_trend'].fillna(0) +
        0.3 * data['final_hour_momentum'].fillna(0)
    )
    
    # DYNAMIC RISK ADJUSTMENT
    # Regime-Adaptive Position Sizing
    volatility_adjustment = np.where(data['volatility_regime'], 0.7, 1.0)
    trend_adjustment = np.where(data['strong_trend'], 1.2, 0.9)
    liquidity_adjustment = np.where(data['liquidity_regime'], 0.6, 1.0)
    
    # FINAL FACTOR CONSTRUCTION
    base_factor = (
        0.5 * momentum_signal +
        0.3 * volume_signal * alignment_score +
        0.2 * intraday_signal
    )
    
    # Apply regime adjustments
    final_factor = base_factor * volatility_adjustment * trend_adjustment * liquidity_adjustment
    
    # Normalize and return
    factor_series = final_factor.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    return factor_series
