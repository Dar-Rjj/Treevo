import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'].shift(-1) - df['open']) / df['open']
    
    # Calculate Volume-Weighted Average Price (VWAP)
    vwap = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    
    # Determine Market Trend
    trend = (df['close'] > df['close'].shift(1)).astype(int) - (df['close'] < df['close'].shift(1)).astype(int)
    
    # Assign weights based on the market trend
    trend_weights = {
        1: {'intraday_range': 0.6, 'close_to_open_return': 0.3, 'vwap': 0.1},
        -1: {'intraday_range': 0.7, 'close_to_open_return': 0.2, 'vwap': 0.1},
        0: {'intraday_range': 0.5, 'close_to_open_return': 0.4, 'vwap': 0.1}
    }
    
    def apply_trend_weights(trend):
        return pd.Series(trend_weights.get(trend, trend_weights[0]))
    
    trend_weights_df = trend.apply(apply_trend_weights)
    
    # Emphasize Recent Data
    recent_factor = (
        0.8 * (intraday_range * trend_weights_df['intraday_range']) +
        0.1 * (close_to_open_return * trend_weights_df['close_to_open_return']) +
        0.1 * (vwap * trend_weights_df['vwap'])
    )
    
    older_factor = (
        0.5 * (intraday_range * trend_weights_df['intraday_range']) +
        0.3 * (close_to_open_return * trend_weights_df['close_to_open_return']) +
        0.2 * (vwap * trend_weights_df['vwap'])
    )
    
    # Final Alpha Factor
    alpha_factor = (recent_factor * 0.7) + (older_factor * 0.3)
    
    return alpha_factor
