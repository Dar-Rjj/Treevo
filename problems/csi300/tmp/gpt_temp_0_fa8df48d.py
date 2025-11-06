import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Microstructure Momentum Factor
    Combines volatility spillover, microstructure momentum, timeframe divergence,
    liquidity regimes, and cross-asset signal integration
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Cross-Asset Volatility Spillover Component
    # Calculate rolling volatility for the stock
    stock_volatility = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    
    # Calculate sector-like volatility using high-low range (proxy for sector ETF)
    sector_volatility = ((data['high'] - data['low']) / data['close']).rolling(window=20, min_periods=10).mean()
    
    # Volatility spillover momentum - how stock volatility responds to sector volatility
    vol_spillover_momentum = (stock_volatility / sector_volatility).pct_change(periods=5)
    
    # 2. Microstructure Momentum Core
    # Order flow momentum using volume and price movement
    price_momentum = data['close'].pct_change(periods=1)
    volume_momentum = data['volume'].pct_change(periods=1)
    
    # Trade size distribution momentum (using amount/volume as proxy for average trade size)
    avg_trade_size = data['amount'] / data['volume'].replace(0, np.nan)
    trade_size_momentum = avg_trade_size.pct_change(periods=3).rolling(window=10, min_periods=5).mean()
    
    # Combine microstructure momentum components
    micro_momentum = (price_momentum * volume_momentum).rolling(window=10, min_periods=5).mean() + trade_size_momentum
    
    # 3. Cross-Timeframe Momentum Divergence
    # Ultra-short momentum (1-3 days)
    ultra_short_momentum = data['close'].pct_change(periods=3).rolling(window=5, min_periods=3).mean()
    
    # Short-term momentum (5-10 days)
    short_term_momentum = data['close'].pct_change(periods=10).rolling(window=10, min_periods=5).mean()
    
    # Medium-term momentum (20 days)
    medium_term_momentum = data['close'].pct_change(periods=20).rolling(window=15, min_periods=10).mean()
    
    # Momentum divergence score
    momentum_divergence = (ultra_short_momentum - short_term_momentum) + (short_term_momentum - medium_term_momentum)
    
    # 4. Liquidity Regime Classification
    # Market depth momentum using volume concentration
    volume_ma_short = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_ma_medium = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_concentration = (volume_ma_short / volume_ma_medium).pct_change(periods=3)
    
    # Spread-width momentum using high-low range as proxy for bid-ask spread
    daily_range = (data['high'] - data['low']) / data['close']
    spread_momentum = daily_range.pct_change(periods=5).rolling(window=10, min_periods=5).mean()
    
    # Liquidity regime score
    liquidity_regime = volume_concentration - spread_momentum
    
    # 5. Cross-Asset Signal Integration
    # Lead-lag momentum using autocorrelation of returns
    returns = data['close'].pct_change()
    lead_lag_momentum = returns.rolling(window=15, min_periods=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Momentum convergence using multiple timeframe alignment
    momentum_alignment = (
        ultra_short_momentum.rolling(window=10, min_periods=5).std() / 
        (medium_term_momentum.rolling(window=10, min_periods=5).std() + 1e-8)
    )
    
    # Final factor combination with weights
    factor = (
        0.25 * vol_spillover_momentum +
        0.30 * micro_momentum +
        0.20 * momentum_divergence +
        0.15 * liquidity_regime +
        0.10 * (lead_lag_momentum - momentum_alignment)
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=60, min_periods=30).mean()) / factor.rolling(window=60, min_periods=30).std()
    
    return factor
