import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Breakout Momentum
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Compute Breakout Signal
    high_breakout = df['high'] / df['high'].shift(1)
    low_breakout = df['low'] / df['low'].shift(1)
    breakout_signal = (high_breakout + low_breakout) / 2
    
    # Volume-Weighted Momentum
    volume_weighted = breakout_signal * df['volume']
    hl_breakout_momentum = volume_weighted / true_range.replace(0, np.nan)
    
    # Volatility Regime Adjusted Return
    # Compute Rolling Volatility
    returns = df['close'].pct_change()
    volatility_20 = returns.rolling(window=20).std()
    
    # Calculate Regime Indicator
    volatility_60_avg = volatility_20.rolling(window=60).mean()
    regime_indicator = np.log(volatility_20 / volatility_60_avg.replace(0, np.nan))
    
    # Adjust Recent Returns
    cum_return_5 = (df['close'] / df['close'].shift(5)) - 1
    volatility_adjusted_return = cum_return_5 * regime_indicator
    
    # Liquidity-Adjusted Price Reversal
    # Compute Price Change
    price_change = df['close'] - df['close'].shift(1)
    
    # Calculate Liquidity Measure
    liquidity = (df['volume'] * df['close']).rolling(window=10).mean()
    
    # Generate Reversal Signal
    reversal_signal = -1 * (price_change / liquidity.replace(0, np.nan))
    
    # Opening Gap Persistence Factor
    # Calculate Opening Gap
    opening_gap = (df['open'] / df['close'].shift(1)) - 1
    
    # Assess Gap Persistence
    intraday_range = (df['high'] - df['low']) / df['open']
    gap_persistence = opening_gap / intraday_range.replace(0, np.nan)
    
    # Volume Confirmation
    volume_ratio = df['volume'] / df['volume'].rolling(window=20).mean()
    gap_factor = gap_persistence * volume_ratio
    
    # Efficiency Ratio Trend Strength
    # Compute Price Efficiency
    daily_returns = df['close'].pct_change()
    abs_sum_returns = abs(daily_returns).rolling(window=10).sum()
    total_change = (df['close'] / df['close'].shift(10)) - 1
    
    # Calculate Efficiency Ratio
    efficiency_ratio = total_change / abs_sum_returns.replace(0, np.nan)
    
    # Combine with Momentum
    momentum_5 = (df['close'] / df['close'].shift(5)) - 1
    efficiency_momentum = momentum_5 * efficiency_ratio
    
    # Volume-Price Divergence Detector
    # Calculate Price Trend
    def linear_trend(series, window):
        x = np.arange(window)
        return series.rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if not y.isna().any() else np.nan, 
            raw=True
        )
    
    price_trend = linear_trend(df['close'], 5)
    volume_trend = linear_trend(df['volume'], 5)
    
    # Generate Divergence Signal
    divergence_signal = np.sign(price_trend * volume_trend)
    
    # Intraday Strength Consistency
    # Compute Intraday Strength
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate Consistency
    consistency = 1 / intraday_strength.rolling(window=10).std()
    
    # Volume Validation
    volume_validation = consistency * (df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Pressure Release Alpha
    # Calculate Buying Pressure
    buying_pressure = ((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)) * df['volume']
    
    # Compute Selling Pressure
    selling_pressure = ((df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)) * df['volume']
    
    # Generate Pressure Differential
    pressure_differential = (buying_pressure - selling_pressure).rolling(window=5).sum()
    
    # Range Expansion Predictor
    # Compute Normalized Range
    normalized_range = (df['high'] - df['low']) / df['close']
    
    # Calculate Range Expansion
    range_expansion = (normalized_range / normalized_range.rolling(window=10).mean()) - 1
    
    # Volume Confirmation
    range_predictor = range_expansion * (df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Momentum Quality Score
    # Calculate Raw Momentum
    raw_momentum = (df['close'] / df['close'].shift(10)) - 1
    
    # Assess Momentum Quality
    up_days = (df['close'] > df['close'].shift(1)).rolling(window=10).sum()
    quality_ratio = up_days / 10
    
    # Combine Signals
    momentum_quality = raw_momentum * quality_ratio
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'hl_breakout': hl_breakout_momentum,
        'vol_adj_return': volatility_adjusted_return,
        'reversal': reversal_signal,
        'gap_factor': gap_factor,
        'efficiency_momentum': efficiency_momentum,
        'divergence': divergence_signal,
        'intraday_strength': volume_validation,
        'pressure': pressure_differential,
        'range_predictor': range_predictor,
        'momentum_quality': momentum_quality
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=60).mean()) / x.rolling(window=60).std())
    
    # Equal-weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
