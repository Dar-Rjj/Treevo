import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term and Long-Term Momentum
    short_term_momentum = df['close'].rolling(window=5).mean()
    long_term_momentum = df['close'].rolling(window=20).mean()
    
    # Create a Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Intraday Momentum Components
    high_low_diff = df['high'] - df['low']
    open_close_momentum = df['close'] - df['open']
    
    # Combine Intraday Momentum Components
    intraday_momentum_avg = (high_low_diff + open_close_momentum) / 2
    intraday_volatility = df['close'].rolling(window=20).std()
    adjusted_intraday_momentum = intraday_momentum_avg / intraday_volatility
    
    # Volume-Weighted Intraday Momentum
    volume_weighted_intraday_momentum = adjusted_intraday_momentum * df['volume']
    
    # Final Integrated Momentum Differential
    integrated_momentum_differential = (momentum_differential * volume_weighted_intraday_momentum) + volume_weighted_intraday_momentum
    
    # Incorporate 25-Day Momentum
    twenty_five_day_momentum = (df['close'] / df['close'].shift(25)) - 1
    
    # Scale by Amount Influence
    amount_25d_avg = df['amount'].rolling(window=25).mean()
    scaled_amount_influence = (df['amount'] / amount_25d_avg)
    
    # Calculate Intraday Return Spread
    intraday_return_spread = (df['high'] - df['low']) / df['open']
    
    # Incorporate Intraday Volume Dynamics and Momentum
    volume_increase_rate = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    price_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    combined_intraday_metrics = (intraday_return_spread + volume_increase_rate + price_momentum) / 3
    
    # Introduce Weighted Moving Averages
    wma_5 = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    wma_20 = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Calculate Weighted Momentum Differentials
    weighted_momentum_differentials = wma_20 - wma_5
    
    # Incorporate Intraday Range Analysis
    intraday_range_ratio = (df['high'] - df['low']) / (df['high'] + df['low'])
    intraday_volume_scaled = intraday_range_ratio * df['volume']
    
    # Synthesize Integrated Factor
    integrated_factor = (integrated_momentum_differential + combined_intraday_metrics + weighted_momentum_differentials + intraday_volume_scaled) / 4
    
    # Adjust for Close-to-Open Reversal and Intraday Volatility
    close_to_open_reversal = (df['open'] - df['close']) / df['close']
    reversal_adjusted_momentum = high_low_diff * close_to_open_reversal
    intraday_volatility_reversal_adjusted = intraday_volatility * reversal_adjusted_momentum
    
    # Final Combined Factor
    final_combined_factor = (integrated_factor * volume_weighted_intraday_momentum) + intraday_volatility_reversal_adjusted
    
    # Smoothing
    final_combined_factor = final_combined_factor.ewm(span=10, adjust=False).mean()
    
    # Additional Momentum Confirmation
    ten_day_momentum = df['close'].rolling(window=10).mean() - df['close']
    if ten_day_momentum > 0 and short_term_momentum < 0:
        final_combined_factor *= 0.9
    elif ten_day_momentum < 0 and short_term_momentum > 0:
        final_combined_factor *= 1.1
    
    return final_combined_factor
