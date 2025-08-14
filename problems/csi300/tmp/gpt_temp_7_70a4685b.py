import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, benchmark_df, sector_df, shares_outstanding):
    """
    Generate a novel and interpretable alpha factor that incorporates intraday momentum,
    relative strength, and market volatility.
    
    Parameters:
    - df: DataFrame with columns (date, open, high, low, close, amount, volume)
    - benchmark_df: DataFrame with the benchmark index prices (close price indexed by date)
    - sector_df: DataFrame with the sector index prices (close price indexed by date)
    - shares_outstanding: Series with shares outstanding for each stock indexed by date
    
    Returns:
    - Series with the alpha factor values indexed by date
    """
    
    # Calculate Intraday Price Movement
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    df['intraday_movement'] = df['high_low_range'] + df['close_open_diff']
    
    # Incorporate Volume Influence
    df['vol_adj_momentum'] = df['volume'] * df['intraday_movement']
    df['sma_vol_adj_momentum'] = df['vol_adj_momentum'].rolling(window=10).mean()
    
    # Adjust for Market Volatility
    df['daily_return'] = df['close'].pct_change()
    df['market_volatility'] = df['daily_return'].rolling(window=30).std()
    df['adjusted_momentum'] = df['sma_vol_adj_momentum'] - df['market_volatility']
    
    # Incorporate Relative Strength
    lookback_period = 20
    df['stock_performance'] = df['close'].pct_change(periods=lookback_period)
    benchmark_df['benchmark_performance'] = benchmark_df.pct_change(periods=lookback_period)
    df['relative_strength'] = df['stock_performance'] - benchmark_df['benchmark_performance']
    
    # Incorporate Sector Performance
    sector_df['sector_performance'] = sector_df.pct_change(periods=lookback_period)
    df['relative_sector_strength'] = df['stock_performance'] - sector_df['sector_performance']
    
    # Incorporate Liquidity Measures
    df['turnover_ratio'] = df['volume'] / shares_outstanding
    df['liquidity_adjusted_momentum'] = df['adjusted_momentum'] / (df['turnover_ratio'] + 1e-6)
    
    # Final Alpha Factor
    alpha_factor = df['liquidity_adjusted_momentum'] + df['relative_strength'] + df['relative_sector_strength']
    
    return alpha_factor
