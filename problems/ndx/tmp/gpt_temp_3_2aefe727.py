import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Detect Significant Volume Increase
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > 2 * avg_volume_20
    
    # Adjust Price Change by Intraday Volatility
    df['adj_price_change'] = df['price_change'] / df['intraday_range']
    
    # Apply Volume-Weighted Adjustment
    df['weighted_adj_price_change'] = df.apply(
        lambda row: row['adj_price_change'] * 2 if row['volume_spike'] else row['adj_price_change'],
        axis=1
    )
    
    # Accumulate Momentum Score
    df['momentum_score'] = df['weighted_adj_price_change'].rolling(window=28).sum()
    
    # Calculate Short-Term Rate of Change (ROC)
    df['roc_short'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate Long-Term Rate of Change (ROC)
    df['roc_long'] = (df['close'] - df['close'].shift(28)) / df['close'].shift(28)
    
    # Combine Short-Term and Long-Term ROCs
    df['momentum_factor'] = (df['roc_short'] + df['roc_long']) / 2
    
    # Calculate Average True Range (ATR)
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), 
        axis=1
    )
    df['atr_10'] = df['true_range'].rolling(window=10).mean()
    
    # Integrate Cumulative Enhanced Momentum
    df['cumulative_enhanced_momentum'] = df['weighted_adj_price_change'].rolling(window=28).sum()
    
    # Calculate Relative Strength
    df['relative_strength'] = (df['close'] - df['close'].rolling(window=28).min()) / \
                              (df['close'].rolling(window=28).max() - df['close'].rolling(window=28).min())
    
    # Incorporate Trend Reversal Indicator
    df['trend_reversal'] = (df['roc_short'] < 0) & (df['roc_short'].shift(1) > 0) | \
                           (df['roc_short'] > 0) & (df['roc_short'].shift(1) < 0)
    
    # Introduce a New Factor Based on Open and Close Prices
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['std_daily_returns'] = df['daily_return'].rolling(window=22).std()
    
    # Combine Factors into a Composite Alpha Factor
    df['composite_alpha_factor'] = (df['momentum_factor'] + df['cumulative_enhanced_momentum'] + 
                                    df['relative_strength'] + df['atr_10'] + df['trend_reversal'].astype(int)) / 5
    
    return df['composite_alpha_factor'].dropna()
