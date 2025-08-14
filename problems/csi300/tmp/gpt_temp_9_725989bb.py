import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term and Long-Term Momentum
    short_term_momentum = df['close'].rolling(window=7).mean()
    long_term_momentum = df['close'].rolling(window=25).mean()
    
    # Create a Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Intraday Momentum Components
    high_low_diff = df['high'] - df['low']
    open_close_mom = df['close'] - df['open']
    
    # Combine Intraday Momentum Components
    avg_intraday_mom = (high_low_diff + open_close_mom) / 2
    close_std_dev = df['close'].rolling(window=20).std()
    intraday_mom_vol_adj = avg_intraday_mom / close_std_dev
    
    # Volume-Weighted Intraday Momentum
    volume_weighted_intraday_mom = intraday_mom_vol_adj * df['volume']
    
    # Final Integrated Momentum Differential
    integrated_momentum = momentum_differential + volume_weighted_intraday_mom
    
    # Volume Confirmation
    volume_change = df['volume'] / df['volume'].shift(1)
    volume_boost = (volume_change > 1.5).astype(int) * 1.5
    integrated_momentum = integrated_momentum * volume_boost
    
    # Smooth with Exponential Moving Average
    ema_12 = integrated_momentum.ewm(span=12, adjust=False).mean()
    
    # Incorporate 30-Day Momentum
    thirty_day_momentum = df['close'] / df['close'].shift(30) - 1
    integrated_momentum = ema_12 + thirty_day_mom
    
    # Adjust for Amount Influence
    amount_avg_30 = df['amount'].rolling(window=30).mean()
    amount_influence = df['amount'] / amount_avg_30
    integrated_momentum = integrated_momentum * amount_influence
    
    # Incorporate Intraday Volume Dynamics and Momentum
    price_momentum = df['close'] / df['close'].shift(1) - 1
    combined_intraday_metrics = (intraday_mom_vol_adj * volume_change * price_momentum)
    
    # Enhanced Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['close']
    
    # Intraday Volume Change
    volume_percent_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Combine Intraday Metrics
    combined_intraday_metrics = intraday_volatility * volume_percent_change
    
    # Analyze Day-to-Day Momentum Continuation
    day_to_day_mom = df['open'] - df['close'].shift(1)
    last_3_days_mom = (df['close'] - df['close'].shift(3)).fillna(0)
    day_to_day_continuation = day_to_day_mom + last_3_days_mom
    
    # Adjust Factor Value Based on Volume Spikes
    volume_moving_avg = df['volume'].rolling(window=10).mean()
    volume_deviation = (df['volume'] - volume_moving_avg) / volume_moving_avg
    factor_adjustment = volume_deviation * (price_momentum > 0).astype(int) * 2 - (price_momentum < 0).astype(int) * 2
    integrated_momentum = integrated_momentum + factor_adjustment
    
    # Adjust for Close-to-Open Reversal and Intraday Volatility
    close_to_open_reversal = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    reversal_adjusted_momentum = high_low_diff * close_to_open_reversal
    intraday_volatility_and_reversal = intraday_volatility * reversal_adjusted_momentum
    
    # Final Combined Factor
    final_combined_factor = (combined_intraday_metrics + integrated_momentum + intraday_volatility_and_reversal)
    
    # Smoothing
    ema_15 = final_combined_factor.ewm(span=15, adjust=False).mean()
    
    return ema_15
