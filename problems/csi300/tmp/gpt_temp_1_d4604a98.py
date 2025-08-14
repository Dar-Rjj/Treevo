import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Short-Term and Long-Term Momentum
    short_term_momentum = df['close'].rolling(window=5).sum()
    long_term_momentum = df['close'].rolling(window=20).mean()
    
    # Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Intraday Momentum Components
    high_low_diff = df['high'] - df['low']
    open_close_momentum = df['close'] - df['open']
    
    # Combine Intraday Momentum Components
    avg_intraday_momentum = (high_low_diff + open_close_momentum) / 2
    intraday_volatility = df['close'].rolling(window=10).std()
    adjusted_intraday_momentum = avg_intraday_momentum / intraday_volatility
    
    # Volume-Weighted Intraday Momentum
    volume_weighted_intraday_momentum = adjusted_intraday_momentum * df['volume']
    
    # Final Integrated Momentum Differential
    final_integrated_momentum = momentum_differential * volume_weighted_intraday_momentum + volume_weighted_intraday_momentum
    
    # 30-Day Momentum
    thirty_day_momentum = (df['close'] / df['close'].shift(30)) - 1
    
    # Scale by Amount Influence
    amount_30_day_avg = df['amount'].rolling(window=30).mean()
    scaled_amount_influence = df['amount'] / amount_30_day_avg
    
    final_integrated_momentum = final_integrated_momentum * scaled_amount_influence
    
    # Adjust for Close-to-Open Reversal
    close_to_open_reversal = (df['open'] - df['close']) / df['close']
    reversal_adjusted_momentum = final_integrated_momentum * close_to_open_reversal
    
    # Intraday Price Momentum
    weighted_high_low = 0.6 * (df['high'] - df['low'])
    weighted_open_close = 0.4 * (df['close'] - df['open'])
    intraday_price_momentum = weighted_high_low + weighted_open_close
    
    # Volume Weighting and Confirmation
    volume_moving_avg = df['volume'].rolling(window=20).mean()
    significant_volume_increase = (df['volume'] > volume_moving_avg)
    
    confirmed_momentum = intraday_price_momentum * significant_volume_increase * df['volume']
    
    # Adjust for Close-to-Open Reversal and Intraday Volatility
    close_to_open_reversal = (df['open'] - df['close']) / df['close']
    reversal_adjusted_high_low = (df['high'] - df['low']) * close_to_open_reversal
    integrated_intraday_metrics = (reversal_adjusted_high_low * intraday_volatility) + reversal_adjusted_momentum
    
    # Final Combined Factor
    final_combined_factor = final_integrated_momentum * volume_weighted_intraday_momentum + integrated_intraday_metrics
    
    # Smoothing with EMA
    final_combined_factor = final_combined_factor.ewm(span=10, adjust=False).mean()
    
    # Additional Momentum Confirmation
    ten_day_momentum = df['close'].rolling(window=10).mean() - df['close']
    
    adjustment = 1
    if ten_day_momentum > 0 and short_term_momentum < 0:
        adjustment = 0.8
    elif ten_day_momentum < 0 and short_term_momentum > 0:
        adjustment = 1.2
    
    final_combined_factor = final_combined_factor * adjustment
    
    return final_combined_factor
