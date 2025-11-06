import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence with Volume Confirmation
    # Calculate Price Momentum
    short_term_momentum = data['close'].pct_change(periods=5)
    medium_term_momentum = data['close'].pct_change(periods=20)
    
    # Compute Momentum Divergence
    momentum_divergence = (short_term_momentum - medium_term_momentum).abs()
    
    # Volume Confirmation
    volume_avg_5 = data['volume'].rolling(window=5).mean()
    volume_ratio = data['volume'] / volume_avg_5
    factor1 = momentum_divergence * volume_ratio
    
    # Volatility Regime Adjusted Return
    # Calculate Daily Return
    daily_return = data['close'].pct_change()
    
    # Measure Volatility Regime
    price_range_20 = (data['high'].rolling(window=20).max() - 
                     data['low'].rolling(window=20).min()) / data['close'].shift(1)
    
    # Calculate Average True Range over 20 days
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = true_range.rolling(window=20).mean() / data['close'].shift(1)
    
    # Adjust Return by Volatility
    factor2 = (daily_return / atr_20) * price_range_20
    
    # Intraday Strength Persistence
    # Calculate Intraday Strength
    intraday_strength = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Measure Persistence
    intraday_avg_5 = intraday_strength.rolling(window=5).mean()
    intraday_std_5 = intraday_strength.rolling(window=5).std()
    
    # Generate Factor
    factor3 = (intraday_avg_5 / intraday_std_5.replace(0, np.nan)) * intraday_strength
    
    # Volume-Price Correlation Breakout
    # Calculate Rolling Correlation
    def rolling_correlation(x, y, window):
        return x.rolling(window=window).corr(y)
    
    vol_price_corr_10 = rolling_correlation(data['volume'], data['close'], 10)
    
    # Detect Breakout
    corr_avg_20 = vol_price_corr_10.rolling(window=20).mean()
    corr_deviation = vol_price_corr_10 - corr_avg_20
    
    # Combine with Price Action
    factor4 = daily_return * corr_deviation
    
    # Acceleration-Deceleration Indicator
    # Calculate Price Acceleration
    returns_5 = data['close'].pct_change(periods=5)
    price_acceleration = returns_5.diff()
    
    # Volume Acceleration
    volume_changes_5 = data['volume'].pct_change(periods=5)
    volume_acceleration = volume_changes_5.diff()
    
    # Generate Composite Factor
    recent_return = data['close'].pct_change()
    factor5 = price_acceleration * volume_acceleration * np.sign(recent_return)
    
    # Resistance Break with Volume Surge
    # Identify Resistance Level
    resistance_level = data['high'].rolling(window=20).max()
    
    # Check Breakout Condition
    breakout_condition = data['high'] > resistance_level.shift(1)
    volume_avg_20 = data['volume'].rolling(window=20).mean()
    volume_surge = data['volume'] > (volume_avg_20 * 1.5)
    
    # Generate Signal
    breakout_signal = (breakout_condition & volume_surge).astype(int)
    volume_surge_magnitude = data['volume'] / volume_avg_20
    factor6 = breakout_signal * volume_surge_magnitude
    
    # Mean Reversion with Volatility Scaling
    # Calculate Price Deviation
    ma_10 = data['close'].rolling(window=10).mean()
    price_deviation = (data['close'] - ma_10) / ma_10
    
    # Scale by Volatility
    std_10 = data['close'].rolling(window=10).std() / data['close']
    scaled_deviation = price_deviation / std_10.replace(0, np.nan)
    
    # Add Momentum Filter
    momentum_5 = data['close'].pct_change(periods=5)
    factor7 = scaled_deviation * np.sign(momentum_5)
    
    # Liquidity-Adjusted Momentum
    # Calculate Traditional Momentum
    momentum_10 = data['close'].pct_change(periods=10)
    
    # Measure Liquidity Impact
    avg_trade_size = data['amount'] / data['volume'].replace(0, np.nan)
    avg_trade_size_5 = avg_trade_size.rolling(window=5).mean()
    
    # Adjust Momentum
    liquidity_adjusted_momentum = momentum_10 / avg_trade_size_5.replace(0, np.nan)
    factor8 = np.log1p(abs(liquidity_adjusted_momentum)) * np.sign(liquidity_adjusted_momentum)
    
    # Open-to-Close Efficiency Ratio
    # Calculate Intraday Return
    intraday_return = (data['close'] - data['open']) / data['open']
    
    # Measure Price Range
    price_range = (data['high'] - data['low']) / data['open']
    
    # Generate Efficiency Factor
    efficiency_ratio = intraday_return / price_range.replace(0, np.nan)
    factor9 = efficiency_ratio * abs(intraday_return)
    
    # Volume-Weighted Price Level
    # Calculate Volume-Weighted Average Price
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).rolling(window=1).sum() / data['volume'].rolling(window=1).sum()
    
    # Compare to Current Price
    price_deviation_vwap = (data['close'] - vwap) / vwap
    
    # Add Volume Trend
    def linear_slope(series, window):
        x = np.arange(window)
        def calc_slope(y):
            if len(y) == window and not y.isna().any():
                return np.polyfit(x, y, 1)[0]
            return np.nan
        return series.rolling(window=window).apply(calc_slope, raw=False)
    
    volume_slope_5 = linear_slope(data['volume'], 5)
    factor10 = price_deviation_vwap * volume_slope_5
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'f1': factor1, 'f2': factor2, 'f3': factor3, 'f4': factor4, 'f5': factor5,
        'f6': factor6, 'f7': factor7, 'f8': factor8, 'f9': factor9, 'f10': factor10
    })
    
    # Z-score normalize each factor and take simple average
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    composite_factor = normalized_factors.mean(axis=1)
    
    return composite_factor
