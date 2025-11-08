import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price reversal magnitude adjusted by volume clustering,
    intraday range efficiency with volume confirmation, liquidity-adjusted momentum persistence,
    volatility compression breakout with volume expansion, and price-volume divergence in trend context.
    """
    df = df.copy()
    
    # Price Reversal Magnitude Adjusted by Volume Clustering
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Extreme price reversal component
    returns_rolling = df['returns'].rolling(window=20, min_periods=10)
    df['returns_zscore'] = (df['returns'] - returns_rolling.mean()) / returns_rolling.std()
    
    # Flag extreme returns (beyond 2 standard deviations)
    df['extreme_move'] = np.abs(df['returns_zscore']) > 2
    
    # Calculate reversal strength (next period return after extreme move)
    df['next_return'] = df['returns'].shift(-1)
    df['reversal_strength'] = -df['next_return'] * df['extreme_move']
    
    # Volume clustering component
    volume_rolling = df['volume'].rolling(window=20, min_periods=10)
    df['volume_percentile'] = volume_rolling.apply(lambda x: (x.iloc[-1] > x.quantile(0.8)), raw=False)
    df['volume_spike'] = df['volume'] > volume_rolling.mean() * 1.5
    
    # Combine reversal with volume clustering
    df['reversal_volume_adj'] = df['reversal_strength'] * (1 + df['volume_percentile'] * 0.5 + df['volume_spike'] * 0.3)
    
    # Intraday Range Efficiency with Volume Confirmation
    df['intraday_range'] = df['high'] - df['low']
    df['open_close_move'] = np.abs(df['close'] - df['open'])
    df['range_efficiency'] = df['open_close_move'] / (df['intraday_range'] + 1e-8)
    
    # Volume confirmation
    df['volume_trend'] = df['volume'].rolling(window=5).mean() > df['volume'].rolling(window=20).mean()
    df['efficiency_volume_align'] = df['range_efficiency'] * (1 + df['volume_trend'] * 0.2)
    
    # Liquidity-Adjusted Momentum Persistence
    df['momentum_direction'] = np.sign(df['returns'])
    df['momentum_streak'] = (df['momentum_direction'] == df['momentum_direction'].shift(1)).cumsum()
    df['momentum_reset'] = df['momentum_direction'] != df['momentum_direction'].shift(1)
    df['streak_counter'] = df.groupby(df['momentum_reset'].cumsum()).cumcount() + 1
    
    # Liquidity adjustment
    df['volume_momentum_sensitivity'] = df['volume'].rolling(window=10).std() / (df['returns'].rolling(window=10).std() + 1e-8)
    df['liquidity_adjusted_momentum'] = df['streak_counter'] / (1 + df['volume_momentum_sensitivity'])
    
    # Volatility Compression Breakout with Volume Expansion
    df['range_ratio'] = df['intraday_range'] / df['intraday_range'].rolling(window=20).mean()
    df['compression_period'] = df['range_ratio'] < 0.7
    
    # Breakout detection
    df['breakout_magnitude'] = df['intraday_range'] / df['intraday_range'].rolling(window=5).mean()
    df['volume_expansion'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    df['breakout_signal'] = (df['breakout_magnitude'] > 1.3) & (df['volume_expansion'] > 1.2) & df['compression_period'].shift(1)
    
    # Price-Volume Divergence in Trend Context
    df['short_ma'] = df['close'].rolling(window=5).mean()
    df['medium_ma'] = df['close'].rolling(window=20).mean()
    df['short_trend'] = df['short_ma'] > df['short_ma'].shift(1)
    df['medium_trend'] = df['medium_ma'] > df['medium_ma'].shift(1)
    
    # Price-volume correlation
    df['price_volume_corr'] = df['close'].rolling(window=10).corr(df['volume'])
    df['divergence_magnitude'] = np.abs(df['price_volume_corr'])
    
    # Trend context weighting
    df['trend_strength'] = np.abs(df['medium_ma'].pct_change(5))
    df['trend_aware_divergence'] = df['divergence_magnitude'] * df['trend_strength']
    
    # Combine all components with appropriate weights
    df['composite_factor'] = (
        df['reversal_volume_adj'].fillna(0) * 0.25 +
        df['efficiency_volume_align'].fillna(0) * 0.20 +
        df['liquidity_adjusted_momentum'].fillna(0) * 0.20 +
        df['breakout_signal'].fillna(0) * 0.15 +
        df['trend_aware_divergence'].fillna(0) * 0.20
    )
    
    # Normalize the final factor
    factor_series = df['composite_factor']
    factor_series = (factor_series - factor_series.rolling(window=50, min_periods=20).mean()) / factor_series.rolling(window=50, min_periods=20).std()
    
    return factor_series
