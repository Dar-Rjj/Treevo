import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Directional Volume Pressure
    # Calculate price changes
    data['close_change'] = data['close'] - data['close'].shift(1)
    
    # Up-day volume pressure
    data['up_volume'] = np.where(data['close_change'] > 0, data['volume'], 0)
    # Down-day volume pressure
    data['down_volume'] = np.where(data['close_change'] < 0, data['volume'], 0)
    # Neutral volume pressure
    data['neutral_volume'] = np.where(data['close_change'] == 0, data['volume'], 0)
    
    # Volume Pressure Momentum
    data['short_volume_pressure'] = data['up_volume'] - data['up_volume'].shift(3)
    data['medium_volume_pressure'] = data['down_volume'] - data['down_volume'].shift(8)
    
    # Long-term volume balance
    up_volume_15d = data['up_volume'].rolling(window=15, min_periods=5).sum()
    down_volume_15d = data['down_volume'].rolling(window=15, min_periods=5).sum()
    data['long_volume_balance'] = (up_volume_15d - down_volume_15d) / (up_volume_15d + down_volume_15d + 1e-8)
    
    # 2. Price-Volume Divergence Patterns
    # Price and volume momentum
    data['price_momentum_5d'] = data['close'] - data['close'].shift(5)
    data['volume_momentum_5d'] = data['volume'] - data['volume'].shift(5)
    
    # Normalized momentum ranks
    price_rank = data['price_momentum_5d'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    volume_rank = data['volume_momentum_5d'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['divergence_strength'] = price_rank - volume_rank
    
    # Divergence regime identification
    data['bullish_divergence'] = ((data['price_momentum_5d'] < 0) & (data['volume_momentum_5d'] > 0)).astype(int)
    data['bearish_divergence'] = ((data['price_momentum_5d'] > 0) & (data['volume_momentum_5d'] < 0)).astype(int)
    data['confirmation'] = ((data['price_momentum_5d'] * data['volume_momentum_5d']) > 0).astype(int)
    
    # Divergence persistence
    for window in [3, 6, 9]:
        data[f'bullish_persistence_{window}d'] = data['bullish_divergence'].rolling(window=window).sum()
        data[f'bearish_persistence_{window}d'] = data['bearish_divergence'].rolling(window=window).sum()
    
    # 3. Amount-Based Trade Size Analysis
    # Average trade size
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    
    # Large trade concentration
    data['large_trade_concentration'] = data['avg_trade_size'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).quantile(0.8), raw=False
    )
    
    # Trade size momentum
    data['trade_size_momentum'] = data['avg_trade_size'] - data['avg_trade_size'].shift(5)
    
    # Trade size vs price impact
    price_change_abs = abs(data['close'] - data['close'].shift(1))
    data['large_trade_efficiency'] = price_change_abs / (data['avg_trade_size'] + 1e-8)
    
    # Small trade accumulation (bottom 20% trade sizes)
    small_trade_threshold = data['avg_trade_size'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).quantile(0.2), raw=False
    )
    data['small_trade_volume'] = np.where(data['avg_trade_size'] <= small_trade_threshold, data['volume'], 0)
    data['small_trade_accumulation'] = data['small_trade_volume'].rolling(window=5, min_periods=3).sum()
    
    # Trade size divergence
    trade_size_rank = data['trade_size_momentum'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    price_momentum_rank = data['price_momentum_5d'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['trade_size_divergence'] = trade_size_rank - price_momentum_rank
    
    # 4. Generate Multi-Timeframe Divergence Signals
    # Combine volume pressure and divergence components
    volume_pressure_strength = (
        data['short_volume_pressure'].fillna(0) * 0.4 +
        data['medium_volume_pressure'].fillna(0) * 0.3 +
        data['long_volume_balance'].fillna(0) * 0.3
    )
    
    # Weight divergence signals by volume pressure strength
    weighted_divergence = data['divergence_strength'] * volume_pressure_strength
    
    # Adjust for trade size confirmation
    trade_size_confirmation = np.where(
        data['trade_size_divergence'] * data['divergence_strength'] > 0,
        abs(data['trade_size_divergence']),
        0
    )
    
    # Scale by divergence persistence
    persistence_factor = (
        data['bullish_persistence_3d'].fillna(0) * 0.5 +
        data['bullish_persistence_6d'].fillna(0) * 0.3 +
        data['bullish_persistence_9d'].fillna(0) * 0.2 -
        data['bearish_persistence_3d'].fillna(0) * 0.5 -
        data['bearish_persistence_6d'].fillna(0) * 0.3 -
        data['bearish_persistence_9d'].fillna(0) * 0.2
    )
    
    # Apply timeframe-specific weighting
    short_term_signal = (
        data['divergence_strength'].fillna(0) * 0.6 +
        data['short_volume_pressure'].fillna(0) * 0.4
    )
    
    medium_term_signal = (
        data['long_volume_balance'].fillna(0) * 0.5 +
        data['bullish_persistence_6d'].fillna(0) * 0.3 -
        data['bearish_persistence_6d'].fillna(0) * 0.2
    )
    
    long_term_signal = (
        data['small_trade_accumulation'].fillna(0) * 0.4 +
        data['trade_size_divergence'].fillna(0) * 0.3 +
        data['large_trade_efficiency'].fillna(0) * 0.3
    )
    
    # Generate final factor
    final_factor = (
        short_term_signal * 0.4 +
        medium_term_signal * 0.35 +
        long_term_signal * 0.25 +
        weighted_divergence * 0.1 +
        trade_size_confirmation * 0.05 +
        persistence_factor * 0.05
    )
    
    # Clean and return the factor series
    factor_series = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor_series.name = 'multi_timeframe_divergence_factor'
    
    return factor_series
