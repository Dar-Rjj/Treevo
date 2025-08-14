import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Enhanced Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['close']
    
    # Calculate Intraday Volume Change
    volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Combine Intraday Metrics
    intraday_metrics = intraday_volatility * volume_change
    
    # Calculate Short-Term Momentum
    short_term_momentum = df['close'].rolling(window=5).mean() - df['close']
    
    # Calculate Long-Term Momentum
    long_term_momentum = df['close'].rolling(window=20).mean() - df['close']
    
    # Create a Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Incorporate Intraday Volume Dynamics
    volume_increase_rate = df['volume'] / df['volume'].shift(1)
    volume_weighted_intraday_volatility = intraday_volatility * df['volume'] / df['volume'].shift(1)
    combined_intraday_volatility = intraday_volatility + volume_weighted_intraday_volatility
    
    # Compute High-Low Range Momentum
    high_low_range_momentum = (df['high'] - df['low']) - (df['high'].shift(1) - df['low'].shift(1))
    
    # Adjust for Close-to-Open Reversal and Intraday Volatility
    close_to_open_reversal = (df['open'] - df['close']) / df['close']
    reversal_adjusted_momentum = high_low_range_momentum * close_to_open_reversal
    integrated_intraday_volatility = intraday_volatility + reversal_adjusted_momentum
    
    # Final Combined Factor
    final_combined_factor = intraday_metrics * momentum_differential + integrated_intraday_volatility
    
    # Smooth with Exponential Moving Average
    final_combined_factor_ema = final_combined_factor.ewm(span=10, adjust=False).mean()
    
    # Introduce Price-Earnings Ratio Impact
    final_combined_factor_pe = final_combined_factor_ema * df['pe_ratio']
    
    # Consider Market Sentiment
    final_combined_factor_sentiment = final_combined_factor_pe * df['sentiment_index']
    
    # Integrate Sector Performance
    final_combined_factor_sector = final_combined_factor_sentiment * df['sector_return']
    
    return final_combined_factor_sector
