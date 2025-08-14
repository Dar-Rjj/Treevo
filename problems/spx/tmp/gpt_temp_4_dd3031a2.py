import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Calculate Short-Term Price Momentum (20 Days)
    df['short_term_momentum'] = df['daily_price_change'].rolling(window=22).sum()
    
    # Calculate Long-Term Price Momentum (60 Days)
    df['long_term_momentum'] = df['daily_price_change'].rolling(window=58).sum()
    
    # Calculate Volume Thrust (10 Days)
    df['daily_volume_change'] = df['volume'].diff()
    df['volume_thrust'] = df['daily_volume_change'].rolling(window=12).sum()
    
    # Calculate Daily Range
    df['daily_range'] = df['high'] - df['low']
    
    # Calculate Range Momentum (15 Days)
    df['range_momentum'] = df['daily_range'].rolling(window=14).sum()
    
    # High-Low Momentum Factor
    df['avg_high_low_range_5d'] = df['daily_range'].rolling(window=5).mean()
    df['high_low_momentum'] = df['daily_range'] - df['avg_high_low_range_5d'].shift(1)
    
    # Volume Ratio
    df['prev_7_day_avg_volume'] = df['volume'].rolling(window=7).mean().shift(1)
    df['volume_ratio'] = df['volume'] / df['prev_7_day_avg_volume']
    
    # Combine Momentum Factors
    df['combined_momentum'] = (df['short_term_momentum'] - df['long_term_momentum']) * df['volume_ratio']
    
    # Synthesize Integrated Alpha Factor
    df['synthesized_alpha_factor'] = (df['short_term_momentum'] * df['volume_thrust'] * df['range_momentum']) + df['high_low_momentum']
    df['volatility'] = df['daily_price_change'].rolling(window=10).std()
    df['integrated_alpha_factor'] = df['synthesized_alpha_factor'] / df['volatility']
    
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Trend Following Momentum
    N = 30  # Example lookback period
    df['trend_following_momentum'] = (df['close'] - df['close'].shift(N)) / df['close'].shift(N)
    
    # Intraday Movement
    df['intraday_movement'] = df['high'] - df['low']
    
    # Daily Return Weighted by Volume
    df['daily_return_weighted_by_volume'] = df['daily_price_change'] * df['volume'] / df['close'].shift(1)
    
    # Compute Volume Change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Determine High Volume Days Relative to Average
    M = 20  # Example moving average window
    df['high_volume_day'] = df['volume'] > df['volume'].rolling(window=M).mean().shift(1)
    df['high_volume_day'] = df['high_volume_day'].astype(int)
    
    # Trade Amount Per Unit of Price
    df['trade_amount_per_unit_price'] = df['amount'] / df['close']
    
    # Analyze Trade Amount Trends
    P = 20  # Example lookback period for amount
    df['trade_amount_trend'] = (df['amount'] - df['amount'].shift(P)) / df['amount'].shift(P)
    
    # Combine with Price and Volume for Complex Signals
    df['complex_signal'] = (df['amount'] / df['close']) * df['volume_change']
    
    # Correlation between Price Changes and Volume
    Q = 10  # Example lag
    df['price_vol_corr'] = df[['close', 'volume']].pct_change().corr(method='pearson').iloc[::2, ::2]
    df['price_vol_corr'] = df['price_vol_corr'].unstack().loc[:, ('close', 'volume')]
    
    # Intraday Range and Closing Price Interaction
    df['intraday_close_interaction'] = df['intraday_movement'] / df['close']
    
    # Price-Volume Ratio
    df['price_volume_ratio'] = (df['close'] / df['close'].shift(1)) * (df['volume'] / df['volume'].shift(1))
    
    return df['integrated_alpha_factor']
