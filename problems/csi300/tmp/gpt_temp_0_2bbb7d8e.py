import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Liquidity Momentum Factor
    Combines liquidity provision patterns with multi-timeframe momentum signals
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Bid-Ask Spread Proxy
    # Effective spread using High, Low, Close prices
    mid_price = (data['high'] + data['low']) / 2
    data['effective_spread'] = 2 * np.abs(data['close'] - mid_price) / mid_price
    
    # Rolling spread statistics for regime detection
    data['spread_ma_5'] = data['effective_spread'].rolling(window=5, min_periods=3).mean()
    data['spread_std_5'] = data['effective_spread'].rolling(window=5, min_periods=3).std()
    
    # 2. Measure Market Depth Changes
    # Average trade size and concentration
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['trade_size_ma_5'] = data['avg_trade_size'].rolling(window=5, min_periods=3).mean()
    data['trade_size_std_5'] = data['avg_trade_size'].rolling(window=5, min_periods=3).std()
    
    # Trade size concentration ratio (current vs average)
    data['trade_size_concentration'] = data['avg_trade_size'] / data['trade_size_ma_5']
    
    # 3. Calculate Multi-Timeframe Returns
    data['ret_1'] = data['close'].pct_change(1)
    data['ret_3'] = data['close'].pct_change(3)
    data['ret_5'] = data['close'].pct_change(5)
    
    # 4. Liquidity Regime Classification
    # Spread compression indicator (lower spread = better liquidity)
    data['spread_compression'] = data['spread_ma_5'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
    )
    
    # Liquidity regime score (0=low liquidity, 1=high liquidity)
    data['liquidity_regime'] = 1 - data['spread_compression']
    
    # 5. Regime-Adaptive Momentum Calculation
    # Weight momentum by liquidity conditions
    data['momentum_1_adj'] = data['ret_1'] * data['liquidity_regime']
    data['momentum_3_adj'] = data['ret_3'] * data['liquidity_regime'].rolling(window=3, min_periods=2).mean()
    data['momentum_5_adj'] = data['ret_5'] * data['liquidity_regime'].rolling(window=5, min_periods=3).mean()
    
    # 6. Market Depth Stability
    # Trade size stability indicator
    data['depth_stability'] = 1 / (1 + data['trade_size_std_5'] / data['trade_size_ma_5'])
    
    # 7. Combine Components into Final Factor
    # Multi-scale momentum weighted by liquidity and depth
    factor = (
        0.4 * data['momentum_1_adj'] +
        0.3 * data['momentum_3_adj'] +
        0.3 * data['momentum_5_adj']
    ) * data['trade_size_concentration'] * data['depth_stability']
    
    # Remove any potential NaN values from early periods
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor
