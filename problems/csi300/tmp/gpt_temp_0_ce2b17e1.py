import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Momentum
    raw_momentum = df['close'] / df['close'].shift(20) - 1
    
    # Calculate 5-Period Moving Average of Returns
    one_period_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    ma_5_returns = one_period_return.rolling(window=5).mean()
    
    # Generate Base Alpha Factor
    high_low_range = df['high'] - df['low']
    base_alpha_factor = raw_momentum * ma_5_returns * high_low_range
    
    # Confirm with Volume
    volume_delta = df['volume'] - df['volume'].shift(20)
    absolute_volume_delta = volume_delta.abs()
    
    # Calculate Daily Momentum
    daily_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Combine Daily Momentum and 5-Period Moving Average of Returns
    combined_daily_momentum_ma = daily_momentum + ma_5_returns
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap = ((df['high'] + df['low']) / 2 * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Adjust Combined Alpha Factor
    adjusted_combined_alpha_factor = base_alpha_factor * absolute_volume_delta * (vwap - df['close'])
    
    # Calculate True Range
    true_range = pd.concat([df['high'] - df['low'],
                            (df['high'] - df['close'].shift(1)).abs(),
                            (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    
    # Calculate Price Volatility
    log_returns = np.log(df['close'] / df['close'].shift(1))
    price_volatility = log_returns.rolling(window=20).apply(lambda x: np.sum(x**2)).fillna(0)
    
    # Enhance Combined Components
    enhanced_combined_components = base_alpha_factor * true_range * np.sqrt(price_volatility)
    
    # Integrate with Intraday and Trend Adjustments
    intraday_return = (df['high'] - df['low']) / df['low']
    volume_adjusted_intraday_return = intraday_return / df['volume']
    enhanced_price_momentum = enhanced_combined_components * volume_adjusted_intraday_return
    
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    previous_day_intraday_high_low_spread = df['high'].shift(1) - df['low'].shift(1)
    
    # Calculate Intraday Momentum
    intraday_momentum = intraday_high_low_spread - previous_day_intraday_high_low_spread
    
    # Generate Additional Alpha Factor Component
    amount_change = df['amount'] - df['amount'].shift(20)
    additional_alpha_factor_component = ma_5_returns * np.sign(volume_delta + amount_change)
    
    # Combine All Components for Final Alpha Factor
    final_alpha_factor = (adjusted_combined_alpha_factor 
                          + intraday_momentum 
                          + additional_alpha_factor_component 
                          + combined_daily_momentum_ma)
    
    # Apply 5-day Exponential Moving Average
    ema_5 = final_alpha_factor.ewm(span=5, adjust=False).mean()
    
    # Final confirmation with sign of Volume Change
    volume_change = df['volume'] - df['volume'].shift(1)
    final_confirmation = np.sign(volume_change) * np.sign(df['close'] - df['close'].shift(1))
    
    # Introduce a threshold to filter out low signal-to-noise ratio
    threshold = 0.5
    alpha_factor = final_confirmation * ema_5
    alpha_factor[alpha_factor.abs() < threshold] = 0
    
    return alpha_factor
