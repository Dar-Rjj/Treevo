import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Relative Momentum Factor
    Combines relative performance, momentum persistence, volatility adjustment, 
    volume confirmation, and composite relative strength analysis
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily returns
    df['return'] = df['close'].pct_change()
    
    # Calculate sector and market returns (simulated as rolling averages)
    # In practice, these would come from external data sources
    df['sector_return'] = df['return'].rolling(window=5, min_periods=1).mean()
    df['market_return'] = df['return'].rolling(window=10, min_periods=1).mean()
    
    # Relative Performance Calculation
    df['rel_stock_sector'] = (df['return'] / (df['sector_return'] + 1e-8)) - 1
    df['rel_stock_market'] = (df['return'] / (df['market_return'] + 1e-8)) - 1
    df['rel_sector_market'] = (df['sector_return'] / (df['market_return'] + 1e-8)) - 1
    
    # Momentum Persistence Analysis
    # Short-term persistence (3-day sign consistency)
    df['momentum_sign'] = np.sign(df['return'])
    df['persistence_3d'] = df['momentum_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: len(set(x)) == 1 if len(x) == 3 else 0
    )
    
    # Medium-term persistence (5-day sign consistency)
    df['persistence_5d'] = df['momentum_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: len(set(x)) == 1 if len(x) == 5 else 0
    )
    
    # Persistence strength (consecutive same-sign days)
    def consecutive_count(series):
        if len(series) == 0:
            return 0
        current_sign = series.iloc[-1]
        count = 1
        for i in range(len(series)-2, -1, -1):
            if series.iloc[i] == current_sign:
                count += 1
            else:
                break
        return count
    
    df['persistence_strength'] = df['momentum_sign'].rolling(window=10, min_periods=1).apply(
        consecutive_count, raw=False
    )
    
    # Volatility-Adjusted Momentum
    df['stock_volatility'] = df['high'] - df['low']
    df['sector_volatility'] = df['stock_volatility'].rolling(window=5, min_periods=1).mean()
    
    # Momentum-to-volatility ratios
    df['mom_vol_ratio_stock'] = df['return'] / (df['stock_volatility'] + 1e-8)
    df['mom_vol_ratio_sector'] = df['sector_return'] / (df['sector_volatility'] + 1e-8)
    
    # Volume Confirmation Patterns
    df['volume_ma'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # High-volume breakout: positive return with volume spike
    df['high_volume_breakout'] = ((df['return'] > 0) & (df['volume_ratio'] > 1.5)).astype(int)
    
    # Low-volume pullback: negative return with volume contraction
    df['low_volume_pullback'] = ((df['return'] < 0) & (df['volume_ratio'] < 0.7)).astype(int)
    
    # Volume divergence: price-volume direction mismatch
    df['volume_divergence'] = ((df['return'] > 0) & (df['volume_ratio'] < 0.8)) | \
                             ((df['return'] < 0) & (df['volume_ratio'] > 1.2))
    df['volume_divergence'] = df['volume_divergence'].astype(int)
    
    # Composite Relative Strength
    # Strong outperformer score
    df['strong_outperformer'] = (
        (df['rel_stock_sector'] > 0) * 0.3 +
        (df['rel_stock_market'] > 0) * 0.3 +
        (df['persistence_strength'] >= 3) * 0.2 +
        (df['high_volume_breakout'] == 1) * 0.2
    )
    
    # Weak underperformer score
    df['weak_underperformer'] = (
        (df['rel_stock_sector'] < 0) * 0.3 +
        (df['rel_stock_market'] < 0) * 0.3 +
        (df['persistence_strength'] <= 2) * 0.2 +
        (df['volume_divergence'] == 1) * 0.2
    )
    
    # Regime detection based on market conditions
    df['market_regime'] = np.where(
        df['market_return'].rolling(window=5, min_periods=1).mean() > 0, 1, -1
    )
    
    # Final composite factor
    # Positive for strong outperformers in bullish regimes
    # Negative for weak underperformers in bearish regimes
    result = (
        df['strong_outperformer'] * (df['market_regime'] == 1) -
        df['weak_underperformer'] * (df['market_regime'] == -1) +
        df['mom_vol_ratio_stock'] * 0.1 +
        df['rel_stock_sector'] * 0.2
    )
    
    # Clean up intermediate columns
    cols_to_drop = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'amount', 'volume']]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return result
