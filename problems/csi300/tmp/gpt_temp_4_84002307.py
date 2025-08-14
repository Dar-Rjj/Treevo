import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Enhanced Price Momentum
    ema_5_close = df['close'].ewm(span=5, adjust=False).mean()
    ema_20_close = df['close'].ewm(span=20, adjust=False).mean()
    ema_5_high = df['high'].ewm(span=5, adjust=False).mean()
    ema_20_high = df['high'].ewm(span=20, adjust=False).mean()
    ema_5_low = df['low'].ewm(span=5, adjust=False).mean()
    ema_20_low = df['low'].ewm(span=20, adjust=False).mean()

    close_momentum = ema_5_close - ema_20_close
    high_momentum = ema_5_high - ema_20_high
    low_momentum = ema_5_low - ema_20_low

    combined_momentum = (close_momentum + (high_momentum - low_momentum)) / 2

    # Adjust for Volume
    ema_5_vol = df['volume'].ewm(span=5, adjust=False).mean()
    ema_20_vol = df['volume'].ewm(span=20, adjust=False).mean()

    volume_adjustment = 1.2 if ema_5_vol > ema_20_vol else 0.8
    adjusted_momentum = combined_momentum * volume_adjustment

    # Adjust Momentum by ATR and Volume Change
    true_range = pd.DataFrame({
        'tr1': df['high'] - df['low'],
        'tr2': (df['high'] - df['close'].shift(1)).abs(),
        'tr3': (df['low'] - df['close'].shift(1)).abs()
    }).max(axis=1)
    atr = true_range.ewm(span=14, adjust=False).mean()

    daily_volume_change = df['volume'] / df['volume'].shift(1)
    adjusted_momentum /= atr
    adjusted_momentum *= daily_volume_change

    # Incorporate Enhanced Price Gaps
    open_to_close_gap = df['open'] - df['close'].shift(1)
    high_to_low_gap = df['high'] - df['low']
    gap_adjusted_momentum = adjusted_momentum + open_to_close_gap + high_to_low_gap

    # Detect Volume Spikes
    volume_spike = df['volume'] - df['volume'].shift(1)
    volume_spike_adjusted_momentum = gap_adjusted_momentum + volume_spike

    # Measure Volume Impact
    ema_15_vol = df['volume'].ewm(span=15, adjust=False).mean()

    # Combine Momentum and Close-to-Low Distance
    close_to_low_distance = df['close'] - df['low']
    momentum_with_distance = volume_spike_adjusted_momentum * close_to_low_distance

    # Incorporate Enhanced Price Reversal Sensitivity
    high_low_spread = df['high'] - df['low']
    open_close_spread = df['open'] - df['close']
    weighted_high_low = high_low_spread * df['volume']
    weighted_open_close = open_close_spread * df['volume']
    weighted_spreads = weighted_high_low + weighted_open_close

    smoothed_high_low_spread = high_low_spread.ewm(span=5, adjust=False).mean()
    enhanced_reversal_sensitivity = weighted_spreads + smoothed_high_low_spread

    # Final Alpha Factor
    alpha_factor = (momentum_with_distance - enhanced_reversal_sensitivity) / ema_15_vol

    return alpha_factor
