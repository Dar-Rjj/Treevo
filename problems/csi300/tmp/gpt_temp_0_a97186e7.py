import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Liquidity Momentum with Regime-Aware Pressure Dynamics
    Combines spread momentum, volume-weighted acceleration, gap persistence, 
    price elasticity, and cross-asset pressure dynamics
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Bid-Ask Spread Momentum Analysis
    # Calculate implied spread from high-low range
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    
    # 3-day spread momentum
    data['spread_momentum'] = (data['spread_proxy'] - data['spread_proxy'].shift(3)) / data['spread_proxy'].shift(3)
    
    # Liquidity regime detection using rolling percentiles
    data['spread_regime'] = data['spread_proxy'].rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > np.percentile(x, 70) else (-1 if x.iloc[-1] < np.percentile(x, 30) else 0),
        raw=False
    )
    
    # 2. Volume-Weighted Price Acceleration
    # Calculate VWAP
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    
    # VWAP momentum (5-day)
    data['vwap_momentum'] = (data['vwap'] - data['vwap'].shift(5)) / data['vwap'].shift(5)
    
    # Volume-weighted close momentum
    data['volume_weighted_close'] = data['close'] * data['volume']
    data['vw_close_momentum'] = (data['volume_weighted_close'] - data['volume_weighted_close'].shift(5)) / data['volume_weighted_close'].shift(5)
    
    # Volume-price divergence
    data['price_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['volume_divergence'] = data['price_momentum'] - data['vw_close_momentum']
    
    # 3. Opening Gap Persistence with Intraday Flow
    # Overnight gap
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap fill ratio (avoid division by zero)
    gap_denominator = data['open'] - data['close'].shift(1)
    data['gap_fill_ratio'] = np.where(
        abs(gap_denominator) > 1e-8,
        (data['close'] - data['open']) / gap_denominator,
        0
    )
    
    # Gap persistence signal
    data['gap_persistence'] = np.where(
        (data['overnight_gap'] * data['gap_fill_ratio']) > 0,  # Same direction
        abs(data['overnight_gap']),
        0
    )
    
    # 4. Price Elasticity with Momentum Regimes
    # Daily price elasticity
    high_low_range = data['high'] - data['low']
    data['price_elasticity'] = np.where(
        high_low_range > 1e-8,
        abs(data['close'] - data['open']) / high_low_range,
        0.5  # Default when range is too small
    )
    
    # 3-day elasticity momentum
    data['elasticity_momentum'] = (data['price_elasticity'] - data['price_elasticity'].shift(3)) / data['price_elasticity'].shift(3)
    
    # Elasticity regime detection
    data['elasticity_regime'] = data['price_elasticity'].rolling(window=15, min_periods=8).apply(
        lambda x: 1 if x.iloc[-1] > np.percentile(x, 70) else (-1 if x.iloc[-1] < np.percentile(x, 30) else 0),
        raw=False
    )
    
    # 5. Cross-Asset Pressure Dynamics (using sector proxy from market data)
    # Relative strength calculation (using rolling market comparison)
    market_returns = data['close'].pct_change(periods=5)
    stock_returns = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['relative_strength'] = stock_returns - market_returns.rolling(window=10).mean()
    
    # Multi-asset pressure alignment
    data['momentum_alignment'] = (
        np.sign(data['vwap_momentum']) * 
        np.sign(data['relative_strength']) * 
        np.sign(data['price_momentum'])
    )
    
    # Final alpha factor construction
    # Combine components with regime-aware weighting
    data['alpha_factor'] = (
        # Spread momentum component (negative weight - compression is bullish)
        -0.3 * data['spread_momentum'].fillna(0) * data['spread_regime'].fillna(0) +
        
        # Volume-weighted acceleration
        0.4 * data['vwap_momentum'].fillna(0) * (1 - abs(data['volume_divergence'].fillna(0))) +
        
        # Gap persistence with volume confirmation
        0.25 * data['gap_persistence'].fillna(0) * np.sign(data['overnight_gap'].fillna(0)) *
        np.tanh(data['volume'].fillna(0) / data['volume'].rolling(window=20).mean().fillna(1)) +
        
        # Elasticity regime momentum
        0.2 * data['elasticity_momentum'].fillna(0) * data['elasticity_regime'].fillna(0) +
        
        # Cross-asset pressure alignment
        0.15 * data['momentum_alignment'].fillna(0) * data['relative_strength'].fillna(0)
    )
    
    # Apply final smoothing and normalization
    alpha_series = data['alpha_factor'].rolling(window=5, min_periods=3).mean()
    
    # Remove any potential lookahead bias by ensuring no future data
    return alpha_series
