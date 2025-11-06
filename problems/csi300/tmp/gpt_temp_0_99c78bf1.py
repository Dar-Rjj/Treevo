import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Volatility Asymmetry Detection
    # Upside volatility (positive returns only)
    upside_vol = df['returns'].rolling(window=15, min_periods=5).apply(
        lambda x: x[x > 0].std() if len(x[x > 0]) >= 3 else np.nan, raw=True
    )
    upside_count = df['returns'].rolling(window=15, min_periods=5).apply(
        lambda x: len(x[x > 0]), raw=True
    )
    
    # Downside volatility (negative returns only)
    downside_vol = df['returns'].rolling(window=15, min_periods=5).apply(
        lambda x: x[x < 0].std() if len(x[x < 0]) >= 3 else np.nan, raw=True
    )
    downside_count = df['returns'].rolling(window=15, min_periods=5).apply(
        lambda x: len(x[x < 0]), raw=True
    )
    
    # Volatility Skew Ratio
    vol_skew = upside_vol / downside_vol
    vol_skew = vol_skew.replace([np.inf, -np.inf], np.nan)
    vol_skew_tanh = np.tanh(vol_skew)
    
    # Price Response Asymmetry
    # Calculate rolling volatility (15-day)
    rolling_vol = df['returns'].rolling(window=15, min_periods=10).std()
    
    # High volatility response (volatility > 80th percentile)
    vol_threshold_high = rolling_vol.rolling(window=30, min_periods=20).quantile(0.8)
    high_vol_mask = rolling_vol > vol_threshold_high
    
    # Low volatility response (volatility < 20th percentile)
    vol_threshold_low = rolling_vol.rolling(window=30, min_periods=20).quantile(0.2)
    low_vol_mask = rolling_vol < vol_threshold_low
    
    # Calculate next day returns for high/low volatility regimes
    high_vol_response = df['returns'].shift(-1).where(high_vol_mask.shift(1)).rolling(window=10, min_periods=5).mean()
    low_vol_response = df['returns'].shift(-1).where(low_vol_mask.shift(1)).rolling(window=10, min_periods=5).mean()
    
    # Response differential
    response_diff = high_vol_response - low_vol_response
    
    # Combine with current regime and volatility skew
    current_vol = rolling_vol
    recent_trend = df['returns'].rolling(window=5, min_periods=3).mean()
    
    # Asymmetric signal generation
    asymmetric_signal = (vol_skew_tanh * response_diff * current_vol * np.sign(recent_trend))
    
    # Volume-Price Efficiency Factor
    # Intraday range efficiency
    intraday_range = (df['high'] - df['low']) / df['close']
    volume_adjusted_range = intraday_range / (df['volume'].rolling(window=5, min_periods=3).mean())
    
    # Price discovery efficiency (how quickly price incorporates volume)
    volume_returns_corr = df['volume'].rolling(window=10, min_periods=5).corr(df['returns'].abs())
    efficiency_metric = volume_returns_corr * volume_adjusted_range
    
    # Inefficiency detection using volume autocorrelation
    volume_autocorr = df['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=True
    )
    
    # Price-volume divergence
    price_volume_divergence = (df['returns'].abs() - df['volume'].pct_change().abs()).rolling(window=5, min_periods=3).mean()
    
    # Efficiency-based signal with mean reversion
    momentum_filter = df['returns'].rolling(window=3, min_periods=2).mean()
    efficiency_signal = ((1 - efficiency_metric) * price_volume_divergence * 
                        np.sign(-momentum_filter) * (1 - volume_autocorr.abs()))
    
    # Regime-Dependent Liquidity Momentum
    # Liquidity regime classification using volume percentiles
    volume_percentile = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
    )
    
    high_liquidity = volume_percentile > 0.7
    low_liquidity = volume_percentile < 0.3
    
    # Momentum calculations
    mom_3day = df['close'].pct_change(3)
    mom_10day = df['close'].pct_change(10)
    
    # Regime-specific momentum weighting
    high_liquidity_signal = mom_3day.where(high_liquidity, 0) + 0.7 * mom_10day.where(high_liquidity, 0)
    low_liquidity_signal = -mom_3day.where(low_liquidity, 0) - 0.3 * mom_10day.where(low_liquidity, 0)
    
    # Smooth regime transitions
    regime_smooth = volume_percentile.rolling(window=5, min_periods=3).mean()
    liquidity_signal = (regime_smooth * high_liquidity_signal + 
                       (1 - regime_smooth) * low_liquidity_signal)
    
    # Price-Level Dependent Reversal
    # Distance to psychological price levels
    current_price = df['close']
    round_levels = np.round(current_price / 10) * 10  # Nearest 10 level
    distance_to_level = (current_price - round_levels) / current_price
    
    # Historical support/resistance using rolling highs/lows
    rolling_high = df['high'].rolling(window=20, min_periods=10).max()
    rolling_low = df['low'].rolling(window=20, min_periods=10).min()
    
    # Proximity to recent extremes
    proximity_to_high = (rolling_high - current_price) / current_price
    proximity_to_low = (current_price - rolling_low) / current_price
    
    # Reversal signal based on price levels
    level_signal = (-np.sign(distance_to_level) * np.minimum(proximity_to_high, proximity_to_low) * 
                   np.sign(-mom_3day))
    
    # Combine all factors with equal weighting
    combined_signal = (
        0.25 * asymmetric_signal + 
        0.25 * efficiency_signal + 
        0.25 * liquidity_signal + 
        0.25 * level_signal
    )
    
    # Normalize the final signal
    final_signal = (combined_signal - combined_signal.rolling(window=20, min_periods=10).mean()) / combined_signal.rolling(window=20, min_periods=10).std()
    
    return final_signal
