import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Asset Liquidity-Regime Microstructure Momentum factor
    """
    # Single-Asset Liquidity Components
    df['bid_ask_spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['volume_concentration'] = df['volume'] / df['volume'].rolling(window=5, min_periods=3).mean()
    df['price_impact'] = (df['close'] - df['open']) / (df['amount'] / (df['volume'] + 1e-8))
    
    # Multi-Timeframe Price Efficiency Momentum
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['efficiency_trend'] = df['intraday_efficiency'].rolling(window=5, min_periods=3).mean()
    df['efficiency_acceleration'] = df['efficiency_trend'].diff()
    
    # Volume-Weighted Price Momentum
    df['volume_weighted_return'] = (df['close'] - df['open']) * df['volume']
    df['volume_momentum'] = df['volume_weighted_return'].rolling(window=3, min_periods=2).sum()
    df['volume_momentum_acceleration'] = df['volume_momentum'].diff()
    
    # Amount-Based Momentum Signals
    df['amount_efficiency'] = (df['close'] - df['open']) / (df['amount'] + 1e-8)
    df['amount_momentum'] = df['amount_efficiency'].rolling(window=5, min_periods=3).sum()
    
    # Liquidity Regime Classification
    df['liquidity_score'] = (
        df['bid_ask_spread_proxy'].rolling(window=20, min_periods=10).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
        ) +
        df['volume_concentration'].rolling(window=20, min_periods=10).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
        ) +
        df['price_impact'].rolling(window=20, min_periods=10).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
        )
    ) / 3
    
    # Liquidity vs Efficiency Divergence
    df['liquidity_efficiency_divergence'] = (
        df['liquidity_score'].diff(3) * df['efficiency_trend'].diff(3) +
        df['liquidity_score'].diff(5) * df['efficiency_trend'].diff(5)
    )
    
    # Cross-Asset Liquidity-Momentum Integration (using rolling correlations as proxy)
    df['cross_asset_volume_confirmation'] = (
        df['volume_momentum'].rolling(window=5, min_periods=3).corr(df['amount_momentum'])
    )
    
    # Liquidity-Regime Adaptive Signal Generation
    high_liquidity_mask = df['liquidity_score'] > df['liquidity_score'].rolling(window=20, min_periods=10).quantile(0.7)
    low_liquidity_mask = df['liquidity_score'] < df['liquidity_score'].rolling(window=20, min_periods=10).quantile(0.3)
    
    # High Liquidity Regime Signals
    high_liquidity_signal = (
        df['efficiency_acceleration'] * df['volume_concentration'] +
        df['liquidity_efficiency_divergence'] * df['cross_asset_volume_confirmation']
    )
    
    # Low Liquidity Regime Signals
    low_liquidity_signal = (
        df['price_impact'] * df['efficiency_trend'].rolling(window=10, min_periods=5).std() +
        df['amount_momentum'] * df['cross_asset_volume_confirmation']
    )
    
    # Multi-Horizon Signal Integration
    immediate_signal = (
        df['liquidity_score'].rolling(window=3, min_periods=2).mean() * 
        df['efficiency_acceleration'] +
        df['liquidity_efficiency_divergence'] * df['volume_concentration']
    )
    
    short_term_signal = (
        df['volume_momentum'] * df['efficiency_trend'].rolling(window=5, min_periods=3).std() +
        df['liquidity_efficiency_divergence'] * df['amount_momentum']
    )
    
    medium_term_signal = (
        df['liquidity_efficiency_divergence'].rolling(window=10, min_periods=5).mean() *
        df['efficiency_trend'].rolling(window=10, min_periods=5).std() *
        df['cross_asset_volume_confirmation']
    )
    
    # Final Factor Calculation with Regime Adaptation
    factor = pd.Series(index=df.index, dtype=float)
    
    # Apply regime-specific weighting
    factor[high_liquidity_mask] = (
        0.4 * high_liquidity_signal[high_liquidity_mask] +
        0.3 * immediate_signal[high_liquidity_mask] +
        0.2 * short_term_signal[high_liquidity_mask] +
        0.1 * medium_term_signal[high_liquidity_mask]
    )
    
    factor[low_liquidity_mask] = (
        0.4 * low_liquidity_signal[low_liquidity_mask] +
        0.3 * immediate_signal[low_liquidity_mask] +
        0.2 * short_term_signal[low_liquidity_mask] +
        0.1 * medium_term_signal[low_liquidity_mask]
    )
    
    # Neutral regime (middle 40%)
    neutral_mask = ~(high_liquidity_mask | low_liquidity_mask)
    factor[neutral_mask] = (
        0.25 * high_liquidity_signal[neutral_mask] +
        0.25 * low_liquidity_signal[neutral_mask] +
        0.25 * immediate_signal[neutral_mask] +
        0.15 * short_term_signal[neutral_mask] +
        0.10 * medium_term_signal[neutral_mask]
    )
    
    # Final normalization
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / (factor.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return factor
