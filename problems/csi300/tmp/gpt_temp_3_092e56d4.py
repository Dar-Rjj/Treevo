import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    high_low_range = df['high'] - df['low']
    intraday_momentum = (df['close'] - df['open']) / df['open']

    # Calculate Volume-Weighted Intraday Return
    volume_weighted_return = (df['high'] * df['volume'] - df['low'] * df['volume']) / df['volume']

    # Combine Intraday Momentum and Volume-Weighted Intraday Return
    combined_momentum = intraday_momentum * volume_weighted_return

    # Calculate Adjusted Reversal Indicator
    lagged_volume_weighted_return = volume_weighted_return.shift(1)
    adjusted_reversal_indicator = df['close'].shift(-1) - lagged_volume_weighted_return

    # Introduce Volume-Weighted Moving Average
    df['vwma_10'] = (df['close'] * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    vwma_deviation = df['close'] - df['vwma_10']

    # Integrate Price Trend
    sma_5 = df['close'].rolling(window=5).mean()
    price_trend = df['close'] - sma_5

    # Calculate Volume-Weighted Average True Range (ATR)
    df['tr'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['low'] - df['close'].shift(1)]].abs().max(axis=1)
    df['vwap_tr'] = (df['tr'] * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()

    # Final Alpha Factor
    final_alpha_factor = (
        combined_momentum +
        adjusted_reversal_indicator +
        vwma_deviation +
        price_trend +
        df['vwap_tr']
    )

    return final_alpha_factor
