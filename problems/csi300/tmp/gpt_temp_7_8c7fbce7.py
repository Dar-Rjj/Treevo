import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Momentum Divergence
    # Calculate volatilities
    df['short_vol_range'] = (df['high'] - df['low']).rolling(window=5).mean()
    df['short_vol_std'] = df['close'].pct_change().rolling(window=5).std()
    df['long_vol_range'] = (df['high'] - df['low']).rolling(window=20).mean()
    df['long_vol_std'] = df['close'].pct_change().rolling(window=20).std()
    
    # Calculate momentum signals
    df['short_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['long_momentum'] = df['close'] / df['close'].shift(20) - 1
    
    # Volatility regime classification and momentum weighting
    vol_ratio = (df['short_vol_range'] + df['short_vol_std']) / (df['long_vol_range'] + df['long_vol_std'])
    regime_momentum = np.where(vol_ratio > 1, df['short_momentum'], df['long_momentum'])
    volatility_divergence = regime_momentum * vol_ratio
    
    # Intraday Pressure Accumulation
    # Buying and selling pressure
    df['buying_pressure'] = np.where(df['close'] > df['open'], 
                                    (df['close'] - df['open']) * df['volume'], 0)
    df['selling_pressure'] = np.where(df['close'] < df['open'], 
                                     (df['open'] - df['close']) * df['volume'], 0)
    
    # Net pressure accumulation
    df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
    df['cumulative_pressure'] = df['net_pressure'].rolling(window=10).sum()
    df['normalized_pressure'] = df['cumulative_pressure'] / df['close'].rolling(window=10).mean()
    
    # Liquidity-Adjusted Trend Persistence
    # Trend strength using linear regression slope
    def linear_slope(series):
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0] if len(series) == 15 else np.nan
    
    df['price_slope'] = df['close'].rolling(window=15).apply(linear_slope, raw=False)
    
    # Trend consistency (consecutive same direction days)
    df['price_change'] = df['close'].diff()
    df['trend_direction'] = np.sign(df['price_change'])
    df['consecutive_trend'] = df['trend_direction'].groupby(
        (df['trend_direction'] != df['trend_direction'].shift()).cumsum()
    ).cumcount() + 1
    
    # Liquidity measures
    df['volume_trend'] = df['volume'].rolling(window=5).apply(linear_slope, raw=False)
    df['spread_estimate'] = (df['high'] - df['low']) / df['close']
    df['spread_trend'] = df['spread_estimate'].rolling(window=5).apply(linear_slope, raw=False)
    
    # Combine trend and liquidity
    liquidity_score = (df['volume_trend'] - df['spread_trend']).fillna(0)
    trend_persistence = df['price_slope'] * df['consecutive_trend'] * liquidity_score
    
    # Relative Strength Rotation (using market proxy - average of all stocks)
    # Since we don't have sector/market data, use rolling market average as proxy
    market_avg = df['close'].rolling(window=20).mean()
    df['vs_market_5d'] = (df['close'] / df['close'].shift(5)) / (market_avg / market_avg.shift(5)) - 1
    df['vs_market_15d'] = (df['close'] / df['close'].shift(15)) / (market_avg / market_avg.shift(15)) - 1
    df['vs_market_30d'] = (df['close'] / df['close'].shift(30)) / (market_avg / market_avg.shift(30)) - 1
    
    # Rotation score combining multiple timeframes
    rotation_consistency = (df['vs_market_5d'] > 0).astype(int) + \
                          (df['vs_market_15d'] > 0).astype(int) + \
                          (df['vs_market_30d'] > 0).astype(int)
    
    rotation_score = (df['vs_market_5d'] + df['vs_market_15d'] + df['vs_market_30d']) * rotation_consistency
    
    # Final composite factor
    composite_factor = (
        volatility_divergence.fillna(0) * 0.3 +
        df['normalized_pressure'].fillna(0) * 0.3 +
        trend_persistence.fillna(0) * 0.2 +
        rotation_score.fillna(0) * 0.2
    )
    
    return pd.Series(composite_factor, index=df.index)
