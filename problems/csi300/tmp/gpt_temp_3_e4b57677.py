import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame, sector_ratios: pd.DataFrame, macro_indicators: pd.DataFrame) -> pd.Series:
    import numpy as np
    import pandas as pd

    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts
    pos_sum = df['positive_amount_vol'].rolling(window=5).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=5).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the log returns using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=20).std() * np.sqrt(252)

    # Calculate the VWAP
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=20).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=20, adjust=False).mean()

    # Momentum factor
    momentum = df['close'].pct_change(periods=20)

    # Mean reversion factor
    mean_reversion = -df['close'].pct_change(periods=5)

    # Trend indicator using moving average convergence divergence (MACD)
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    trend_indicator = df['macd'] - df['signal_line']

    # Adaptive weighting using a simple moving average of the factor performance
    df['momentum_perf'] = momentum.rolling(window=20).mean()
    df['mean_reversion_perf'] = mean_reversion.rolling(window=20).mean()
    df['trend_perf'] = trend_indicator.rolling(window=20).mean()
    total_perf = df['momentum_perf'] + df['mean_reversion_perf'] + df['trend_perf']
    
    weight_momentum = df['momentum_perf'] / total_perf
    weight_mean_reversion = df['mean_reversion_perf'] / total_perf
    weight_trend = df['trend_perf'] / total_perf

    # Incorporate sector-specific ratios
    df = df.join(sector_ratios, how='left')
    sector_weight = 0.1  # Weight for sector-specific ratios
    sector_factor = df['sector_ratio']  # Example column name for sector-specific ratios

    # Incorporate macroeconomic indicators
    df = df.join(macro_indicators, how='left')
    macro_weight = 0.1  # Weight for macroeconomic indicators
    macro_factor = df['macro_indicator']  # Example column name for macroeconomic indicators

    # Combine factors with dynamic weights
    alpha_factor = (sentiment_factor * 0.3 +
                    (1/volatility) * 0.2 +
                    (df['vwap'] / df['vwap_smoothed']) * 0.2 +
                    momentum * weight_momentum +
                    mean_reversion * weight_mean_reversion +
                    trend_indicator * weight_trend +
                    sector_factor * sector_weight +
                    macro_factor * macro_weight)

    return alpha_factor
