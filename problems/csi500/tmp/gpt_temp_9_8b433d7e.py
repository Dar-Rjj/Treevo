import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, m=20):
    # Calculate Daily Close Price Trend
    close_trend = df['close'].pct_change(periods=n)
    
    # Smooth the Close Price Trend
    ema_close_21 = df['close'].ewm(span=21, adjust=False).mean()
    momentum_score = ema_close_21.pct_change()

    # Calculate Daily Volume Trend
    volume_trend = df['volume'].pct_change(periods=m)
    
    # Smooth the Volume Trend
    ema_volume_21 = df['volume'].ewm(span=21, adjust=False).mean()
    volume_score = ema_volume_21.pct_change()

    # Combine Momentum and Volume Scores
    combined_momentum_volume = momentum_score * volume_score

    # Calculate Intraday Price Move
    intraday_move = df['high'] - df['low']

    # Volume-Adjusted Intraday Move
    volume_adjusted_intraday_move = (df['volume'] / df['close']) * intraday_move

    # Momentum Signal
    ema_short = df['close'].ewm(span=12, adjust=False).mean()
    ema_long = df['close'].ewm(span=26, adjust=False).mean()
    momentum_signal = ema_short - ema_long

    # Combined Intraday-Momentum Factor
    combined_intraday_momentum = volume_adjusted_intraday_move * momentum_signal

    # Intraday Return
    intraday_return = (df['high'] - df['low']) / df['open']

    # Overlap Period Return
    overlap_period_return = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)

    # Volume Growth
    volume_growth = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)

    # Weighted Returns
    weighted_returns = intraday_return * volume_growth + overlap_period_return * (1 - volume_growth)

    # Final Alpha Factor
    final_alpha_factor = combined_intraday_momentum + weighted_returns

    # Additional Factors
    def rsi(series, periods=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def ad_line(high, low, close, volume):
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        return ad_line

    def atr(high, low, close, periods=14):
        tr = np.maximum.reduce([high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1))])
        atr = tr.rolling(window=periods).mean()
        return atr

    def mfi(high, low, close, volume, periods=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=periods).sum()
        negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=periods).sum()
        mfi = 100 - (100 / (1 + (positive_money_flow / negative_money_flow)))
        return mfi

    # Calculate Additional Factors
    rsi_factor = rsi(df['close'])
    ad_line_factor = ad_line(df['high'], df['low'], df['close'], df['volume'])
    atr_factor = atr(df['high'], df['low'], df['close'])
    mfi_factor = mfi(df['high'], df['low'], df['close'], df['volume'])

    # Composite Alpha Factor
    composite_alpha_factor = final_alpha_factor + rsi_factor + ad_line_factor + atr_factor + mfi_factor

    return composite_alpha_factor
