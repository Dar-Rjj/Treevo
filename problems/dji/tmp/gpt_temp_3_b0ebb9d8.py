import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate EMAs and VWMA
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    short_vwma = (df['close'] * df['volume']).rolling(window=12).sum() / df['volume'].rolling(window=12).sum()
    long_vwma = (df['close'] * df['volume']).rolling(window=26).sum() / df['volume'].rolling(window=26).sum()

    # Adjusted Momentum Calculation
    vwma_diff = short_vwma - long_vwma
    adjusted_momentum = vwma_diff * df['volume']

    # Price Velocity and Acceleration
    daily_close_change = df['close'].diff()
    price_acceleration = daily_close_change.diff()

    # Composite Momentum Signal
    ema_diff = short_ema - long_ema
    composite_momentum = (ema_diff * df['volume']) + price_acceleration

    # Volume Shock Adjustment
    volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    volume_shock_adjustment = (adjusted_momentum - composite_momentum) * volume_change

    # Volume Weighted Price Momentum
    vwap = (df['high'] + df['low'] + df['close']) / 3
    weighted_price = vwap * df['volume']
    vwap_sum = weighted_price.rolling(window=7).sum()

    # High-Low Range Volatility
    high_low_range = df['high'] - df['low']
    avg_high_low_range = high_low_range.rolling(window=7).mean()

    # Intermediate Alpha Factor Synthesis
    intermediate_factor = (adjusted_momentum + vwap_sum) + avg_high_low_range

    # Intraday and Open-to-Close Momentum
    intraday_high_low_range = df['high'] - df['low']
    open_close_momentum = df['close'] - df['open']

    # Integrate Intraday and Open-to-Close Momentum
    integrated_momentum = intraday_high_low_range + open_close_momentum

    # Apply Volume Filter
    volume_rolling_mean = df['volume'].rolling(window=10).mean()
    high_volume_days = df['volume'] > volume_rolling_mean
    filtered_intermediate_factor = intermediate_factor[high_volume_days]

    # Adjust Momentum by Volatility
    high_low_volatility = df['high'] - df['low']
    adjusted_alpha_factor = filtered_intermediate_factor / high_low_volatility

    # Incorporate Amount in the Alpha Factor
    amount_rolling_avg = df['amount'].rolling(window=10).mean()
    amount_ratio = df['amount'] / amount_rolling_avg

    # Finalize Alpha Factor
    final_alpha_factor = adjusted_alpha_factor * amount_ratio

    # Price and Volume Correlation
    price_vol_corr = df[['close', 'volume']].rolling(window=10).corr().unstack().iloc[::2]['close']['volume']
    final_alpha_factor += price_vol_corr

    return final_alpha_factor.dropna()
