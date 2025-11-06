import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining microstructure reversal, cross-asset information flow,
    behavioral anchoring patterns, and multi-timeframe regime synthesis.
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Microstructure-Based Reversal
    # Estimate bid-ask spread using high-low range relative to close
    data['spread_estimate'] = (data['high'] - data['low']) / data['close']
    
    # Relative spread volatility (5-day rolling std)
    data['spread_vol_5d'] = data['spread_estimate'].rolling(window=5, min_periods=3).std()
    
    # Spread-return correlation (10-day rolling)
    data['spread_return_corr_10d'] = data['spread_estimate'].rolling(window=10, min_periods=5).corr(
        data['close'].pct_change()
    )
    
    # Trade imbalance using volume and amount
    data['avg_trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['trade_imbalance'] = (data['volume'] * data['avg_trade_size'].shift(1)).rolling(
        window=5, min_periods=3
    ).mean()
    
    # 2. Cross-Asset Information Flow (using sector proxy from market data)
    # Calculate rolling sector proxy as market average
    data['sector_proxy'] = data['close'].rolling(window=20, min_periods=10).mean()
    data['stock_sector_divergence'] = (data['close'] / data['sector_proxy'] - 1)
    
    # Sector-relative momentum
    data['sector_rel_momentum'] = (
        data['close'].pct_change(5) - 
        data['sector_proxy'].pct_change(5)
    )
    
    # 3. Behavioral Anchoring Patterns
    # Round number clustering effect
    data['round_level_distance'] = np.abs((data['close'] % 1) - 0.5) * 2  # Distance from nearest 0.5 level
    
    # Recent high/low proximity effects
    data['recent_high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['recent_low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['high_proximity'] = (data['close'] - data['recent_low_5d']) / (data['recent_high_5d'] - data['recent_low_5d'])
    
    # Gap fill detection
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_fill_signal'] = -np.sign(data['overnight_gap']) * data['overnight_gap'].abs()
    
    # 4. Multi-Timeframe Regime Synthesis
    # Volatility clustering detection
    data['volatility_5d'] = data['close'].pct_change().rolling(window=5, min_periods=3).std()
    data['volatility_regime'] = data['volatility_5d'] > data['volatility_5d'].rolling(window=20, min_periods=10).mean()
    
    # Trend persistence metrics
    data['trend_strength_10d'] = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.std(x) if len(x) > 1 else np.nan
    )
    
    # Adaptive factor weighting based on market regime
    # Short-term mean reversion factor
    short_term_rev = (
        -data['spread_return_corr_10d'].fillna(0) * 0.3 +
        data['gap_fill_signal'].fillna(0) * 0.4 +
        -data['high_proximity'].fillna(0) * 0.3
    )
    
    # Medium-term momentum factor
    medium_term_mom = (
        data['sector_rel_momentum'].fillna(0) * 0.5 +
        data['trend_strength_10d'].fillna(0) * 0.5
    )
    
    # Long-term value factor (using trade imbalance as proxy)
    long_term_value = data['trade_imbalance'].fillna(0)
    
    # Regime-adaptive combination
    volatility_weight = data['volatility_regime'].astype(float)
    
    # High volatility: emphasize mean reversion, low volatility: emphasize momentum
    combined_factor = (
        volatility_weight * short_term_rev +
        (1 - volatility_weight) * medium_term_mom +
        0.2 * long_term_value  # Small constant weight for long-term component
    )
    
    # Final normalization
    factor_series = (combined_factor - combined_factor.rolling(window=20, min_periods=10).mean()) / \
                   combined_factor.rolling(window=20, min_periods=10).std()
    
    return factor_series
