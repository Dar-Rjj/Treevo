import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-Low Spread
    df['intraday_high_low'] = df['high'] - df['low']
    
    # Logarithmic Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volume Weighted Logarithmic Return
    df['vol_weighted_log_return'] = df['log_return'] * df['volume']
    
    # Enhanced Liquidity Measure
    df['enhanced_liquidity'] = df['volume'] / df['amount']
    
    # Market Trend Adjustment
    df['20_day_ma'] = df['close'].rolling(window=20).mean()
    df['market_trend_adj'] = df['close'] - df['20_day_ma']
    
    # Refined Volume-Weighted Momentum (using a 10-day rolling window)
    df['refined_vol_weighted_momentum'] = df['vol_weighted_log_return'].rolling(window=10).mean()
    
    # Combine Intraday Volatility, Momentum, Enhanced Liquidity, and Market Trend Measures
    combined_measures = (
        df['intraday_high_low'] +
        df['refined_vol_weighted_momentum'] +
        df['enhanced_liquidity'] +
        df['market_trend_adj']
    )
    
    # Dynamic Gap Adjustment
    df['gap_diff'] = df.apply(lambda row: row['open'] - df.loc[row.name - pd.Timedelta(days=1), 'close'] if row['open'] > df.loc[row.name - pd.Timedelta(days=1), 'close'] else df.loc[row.name - pd.Timedelta(days=1), 'close'] - row['open'], axis=1)
    df['adjusted_gap_diff'] = df['gap_diff'] * df['volume']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = combined_measures + df['adjusted_gap_diff']
    
    return df['final_alpha_factor']
