import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Momentum-Liquidity Divergence factor
    Combines momentum analysis with liquidity dynamics and volatility adjustment
    to detect divergences between price movement and volume confirmation
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Analysis with Volatility Adjustment
    # Short-term momentum calculations
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Volatility measures
    data['return_vol_20d'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    data['hl_range_vol'] = (data['high'] - data['low']) / data['close']
    data['hl_range_vol_5d'] = data['hl_range_vol'].rolling(window=5, min_periods=3).mean()
    
    # Volatility-adjusted momentum
    data['vol_adj_momentum_5d'] = data['momentum_5d'] / (data['return_vol_20d'] + 1e-8)
    data['vol_adj_momentum_10d'] = data['momentum_10d'] / (data['return_vol_20d'] + 1e-8)
    
    # Momentum trend assessment
    data['momentum_slope_5d'] = data['vol_adj_momentum_5d'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=True
    )
    data['momentum_slope_10d'] = data['vol_adj_momentum_10d'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan, raw=True
    )
    
    # 2. Liquidity Dynamics Analysis
    # Volume acceleration
    data['volume_5d_change'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_10d_change'] = data['volume'] / data['volume'].shift(10) - 1
    
    # Volume trend slopes
    data['volume_slope_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=True
    )
    data['volume_slope_10d'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan, raw=True
    )
    
    # Turnover efficiency
    data['turnover_efficiency'] = data['amount'] / (data['volume'] + 1e-8)
    data['turnover_trend_10d'] = data['turnover_efficiency'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan, raw=True
    )
    
    # Liquidity quality assessment
    data['volume_consistency_10d'] = data['volume'].rolling(window=10, min_periods=5).std() / (
        data['volume'].rolling(window=10, min_periods=5).mean() + 1e-8
    )
    
    # 3. Divergence Detection Framework
    # Momentum-liquidity alignment
    data['momentum_liquidity_alignment'] = (
        np.sign(data['momentum_slope_5d']) * np.sign(data['volume_slope_5d'])
    )
    
    # Volatility-weighted divergence signals
    volatility_weight = 1 + data['hl_range_vol_5d'] * 2  # Higher weight in high volatility
    data['divergence_strength'] = (
        (data['momentum_slope_5d'] - data['volume_slope_5d']) * volatility_weight
    )
    
    # Multi-timeframe divergence
    data['short_term_alignment'] = (
        np.sign(data['momentum_slope_5d']) * np.sign(data['volume_slope_5d'])
    )
    data['medium_term_alignment'] = (
        np.sign(data['momentum_slope_10d']) * np.sign(data['volume_slope_10d'])
    )
    data['multi_timeframe_consistency'] = (
        data['short_term_alignment'] + data['medium_term_alignment']
    )
    
    # 4. Signal Generation & Classification
    # Base divergence factor
    data['base_divergence'] = (
        data['vol_adj_momentum_5d'] * 0.4 +
        data['momentum_slope_5d'] * 0.3 +
        data['volume_slope_5d'] * 0.3
    )
    
    # Volatility adjustment for final signal
    volatility_adjustment = 1 / (1 + data['hl_range_vol_5d'] * 5)  # Reduce weight in high volatility
    
    # Final factor combining all elements
    data['vwmld_factor'] = (
        data['base_divergence'] * 
        data['momentum_liquidity_alignment'] * 
        volatility_adjustment * 
        (1 + data['multi_timeframe_consistency'] * 0.2)
    )
    
    # Clean up intermediate columns
    result = data['vwmld_factor'].copy()
    
    return result
