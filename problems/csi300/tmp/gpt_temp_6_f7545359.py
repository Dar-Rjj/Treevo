import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)
    
    # Recent Weights: 0.7 for Intraday Range, 0.3 for Close-to-Open Return
    # Older Weights: 0.5 for Intraday Range, 0.5 for Close-to-Open Return
    df['recent_weights_combined'] = df['intraday_range'] * 0.7 + df['close_to_open_return'] * 0.3
    df['older_weights_combined'] = df['intraday_range'] * 0.5 + df['close_to_open_return'] * 0.5
    
    # Calculate 5-Day Momentum
    df['5_day_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Integrate Momentum with Combined Return
    df['momentum_integrated_factor'] = (df['recent_weights_combined'] * 0.7) + (df['5_day_momentum'] * 0.3)
    
    # Calculate Volume Change
    df['volume_shock'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Integrate Volume Shock with Momentum Integrated Factor
    df['volume_adjusted_momentum_factor'] = (df['momentum_integrated_factor'] * 0.8) + (df['volume_shock'] * 0.2)
    
    # Calculate 10-Day Moving Average of Close Prices
    df['10_day_ma'] = df['close'].rolling(window=10).mean()
    
    # Integrate Market Sentiment with Volume-Adjusted Momentum Factor
    df['final_factor'] = (df['volume_adjusted_momentum_factor'] * 0.6) + (df['10_day_ma'] * 0.4)
    
    # Evaluate Recent Market Conditions and Apply Dynamic Weights
    # For simplicity, we assume recent market conditions are based on the last 10 days
    recent_volatility = df['close'].pct_change().rolling(window=10).std()
    recent_volume = df['volume'].rolling(window=10).mean()
    recent_sentiment = df['close'].rolling(window=10).mean() / df['close']
    
    # Adjust Weights Dynamically
    dynamic_weight = 0.7 if (recent_volatility > 0.05 or recent_volume > 1.5 * recent_volume.mean() or recent_sentiment > 1.1) else 0.5
    df['final_factor_dynamic'] = (df['volume_adjusted_momentum_factor'] * 0.6 * dynamic_weight) + (df['10_day_ma'] * 0.4 * (1 - dynamic_weight))
    
    return df['final_factor_dynamic'].dropna()
