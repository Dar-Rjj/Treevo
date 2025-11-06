import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate daily returns
    data['daily_return'] = data['close'].pct_change()
    data['prev_close'] = data['close'].shift(1)
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    
    # Relative Strength Component
    # 1. Stock 20-day momentum (proxy for sector-relative since we don't have sector data)
    data['stock_20d_return'] = data['close'] / data['close'].shift(20) - 1
    
    # 2. Industry group ranking proxy (using rolling percentile within same stock)
    data['rolling_10d_return'] = data['close'] / data['close'].shift(10) - 1
    data['industry_rank'] = data['rolling_10d_return'].rolling(window=60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else np.nan, raw=False
    )
    
    # 3. Cross-market correlation breakdown (using rolling correlation with market proxy)
    # Create market proxy as rolling average of all available stocks (simplified)
    market_proxy = data['close'].rolling(window=5).mean()
    data['market_correlation'] = data['close'].rolling(window=5).corr(market_proxy)
    
    # Liquidity Divergence Analysis
    # 1. Volume-to-amount divergence
    data['volume_20d_trend'] = data['volume'].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 10 else np.nan, raw=False
    )
    data['amount_20d_trend'] = data['amount'].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 10 else np.nan, raw=False
    )
    data['volume_amount_divergence'] = data['volume_20d_trend'] - data['amount_20d_trend']
    
    # 2. Bid-ask spread proxy (Daily range normalized)
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['avg_spread_10d'] = data['spread_proxy'].rolling(window=10).mean()
    data['spread_ratio'] = data['spread_proxy'] / data['avg_spread_10d']
    
    # 3. Large trade concentration proxy (using volume distribution)
    data['volume_std_20d'] = data['volume'].rolling(window=20).std()
    data['volume_mean_20d'] = data['volume'].rolling(window=20).mean()
    data['large_trade_concentration'] = (data['volume'] - data['volume_mean_20d']) / data['volume_std_20d']
    
    # Market Microstructure Signals
    # 1. Opening gap persistence
    data['opening_gap'] = (data['open'] - data['prev_close']).abs() / data['prev_close']
    data['prev_range'] = (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(2)
    data['gap_persistence'] = data['opening_gap'] / (data['prev_range'] + 1e-8)
    
    # 2. Intraday reversal pattern (using high-low range vs close-open)
    data['intraday_move'] = (data['close'] - data['open']) / data['open']
    data['reversal_signal'] = -data['intraday_move'] * data['daily_range']
    
    # 3. Closing auction pressure proxy (using last hour volume pattern)
    # Since we don't have intraday data, use daily volume concentration
    data['volume_30d_avg'] = data['volume'].rolling(window=30).mean()
    data['closing_pressure'] = data['volume'] / (data['volume_30d_avg'] + 1e-8)
    
    # Combine all components with appropriate weights
    # Relative Strength (40% weight)
    rel_strength = (
        0.4 * data['stock_20d_return'] +
        0.35 * data['industry_rank'] +
        0.25 * (1 - data['market_correlation'].abs())
    )
    
    # Liquidity Divergence (35% weight)
    liquidity_div = (
        0.4 * data['volume_amount_divergence'] +
        0.35 * (-data['spread_ratio']) +  # Lower spread is better
        0.25 * data['large_trade_concentration']
    )
    
    # Market Microstructure (25% weight)
    microstructure = (
        0.4 * (-data['gap_persistence']) +  # Lower gap persistence is better
        0.35 * data['reversal_signal'] +
        0.25 * data['closing_pressure']
    )
    
    # Final factor combination
    factor = (
        0.4 * rel_strength +
        0.35 * liquidity_div +
        0.25 * microstructure
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=60, min_periods=20).mean()) / factor.rolling(window=60, min_periods=20).std()
    
    return factor
