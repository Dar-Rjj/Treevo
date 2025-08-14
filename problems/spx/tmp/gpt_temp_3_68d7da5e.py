import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate Daily Price Movement
    df['daily_price_movement'] = df['close'] - df['open']
    
    # Adjust for Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Volume-Weighted Price Movement
    df['volume_weighted_price_movement'] = (df['daily_price_movement'] * df['volume']) / df['intraday_volatility']
    
    # Incorporate Transaction Amount Impact
    df['amount_impact'] = (df['amount'] * df['volume_weighted_price_movement']) / df['volume']
    
    # Identify Directional Days
    df['direction'] = 'Up'
    df.loc[df['close'] < df['open'], 'direction'] = 'Down'
    df['up_count'] = df['direction'].rolling(window=n).apply(lambda x: (x == 'Up').sum())
    df['down_count'] = df['direction'].rolling(window=n).apply(lambda x: (x == 'Down').sum())
    
    # Weight by Volume and Amount
    df['volume_amount_weighted_direction'] = (df['up_count'] - df['down_count']) * (df['volume'] + df['amount'])
    
    # Combine Volume-Weighted Price Movement and Directional Counts
    df['combined_volume_weighted'] = df['volume_weighted_price_movement'] + df['volume_amount_weighted_direction']
    
    # Calculate Short-Term Price Momentum
    df['short_term_momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['agg_short_term_momentum'] = df['short_term_momentum'].rolling(window=5).sum()
    
    # Calculate Long-Term Price Momentum
    df['long_term_momentum'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['agg_long_term_momentum'] = df['long_term_momentum'].rolling(window=20).sum()
    
    # Calculate Intraday Momentum
    df['intraday_momentum'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Spike
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['agg_5_day_volume_spike'] = df['volume_change'].rolling(window=5).sum()
    df['agg_10_day_volume_spike'] = df['volume_change'].rolling(window=10).sum()
    
    # Volume Trend Factor
    df['volume_trend'] = df['volume'] - df['volume'].rolling(window=10).mean()
    
    # Weighted Combination
    df['weighted_combination'] = (df['agg_short_term_momentum'] * df['volume_trend']) + df['agg_long_term_momentum']
    
    # Calculate Trading Activity Indicator
    df['trading_activity_indicator'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Adjust for Intraday Volatility
    df['adjusted_intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Volume-Adjusted Price Movement
    df['volume_adjusted_price_movement'] = (df['close'] - df['open']) * df['volume'] / df['adjusted_intraday_volatility']
    
    # Incorporate Transaction Amount Impact
    df['transaction_amount_impact'] = (df['volume_adjusted_price_movement'] * df['amount']) / df['volume']
    
    # Combine Interaction Terms
    df['interaction_terms'] = (
        df['agg_short_term_momentum'] * df['intraday_momentum'] +
        df['intraday_momentum'] * df['volume_trend'] +
        df['agg_short_term_momentum'] * df['agg_5_day_volume_spike'] +
        df['agg_long_term_momentum'] * df['intraday_momentum']
    )
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['combined_volume_weighted'] * (df['volume'] + df['amount'])) + df['interaction_terms']
    
    return df['final_alpha_factor']
