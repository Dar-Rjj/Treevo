import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining multiple heuristics:
    1. Price and Volume Momentum Divergence
    2. Volatility Regime Adjusted Return
    3. Intraday Pressure Accumulation
    4. Liquidity Gap Factor
    5. Multi-Timeframe Trend Consistency
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Price and Volume Momentum Divergence
    # Calculate 5-day price momentum
    price_momentum = data['close'].pct_change(5)
    
    # Calculate 5-day volume momentum
    volume_momentum = data['volume'].pct_change(5)
    
    # Handle zero volume momentum cases
    volume_momentum_safe = volume_momentum.replace(0, np.nan)
    
    # Calculate momentum ratio (price momentum / volume momentum)
    momentum_ratio = price_momentum / volume_momentum_safe
    
    # Generate divergence signal
    # Positive when price momentum > 0 and volume momentum < 0 (increasing price, decreasing volume)
    # Negative when price momentum < 0 and volume momentum > 0 (decreasing price, increasing volume)
    divergence_signal = np.where(
        (price_momentum > 0) & (volume_momentum < 0), momentum_ratio,
        np.where((price_momentum < 0) & (volume_momentum > 0), -momentum_ratio, 0)
    )
    
    # 2. Volatility Regime Adjusted Return
    # Calculate historical volatility using high-low range
    daily_range = (data['high'] - data['low']) / data['close']
    volatility_20d = daily_range.rolling(window=20).std()
    
    # Classify volatility regimes
    vol_threshold = volatility_20d.quantile(0.7)
    volatility_regime = np.where(volatility_20d > vol_threshold, 1, 0)  # 1=high, 0=low
    
    # Calculate raw returns
    returns_1d = data['close'].pct_change()
    
    # Scale returns by regime (amplify in low vol, attenuate in high vol)
    regime_adjusted_returns = returns_1d * (1.5 - 0.5 * volatility_regime)
    
    # 3. Intraday Pressure Accumulation
    # Calculate intraday strength (buy pressure)
    intraday_strength = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    buy_pressure = (intraday_strength * data['volume']).rolling(window=3).sum()
    
    # Calculate intraday weakness (sell pressure)
    intraday_weakness = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    sell_pressure = (intraday_weakness * data['volume']).rolling(window=3).sum()
    
    # Net pressure accumulation
    pressure_accumulation = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-8)
    
    # 4. Liquidity Gap Factor
    # Volume gap detection (unusually low volume)
    volume_ma_20 = data['volume'].rolling(window=20).mean()
    volume_gap = data['volume'] < (volume_ma_20 * 0.6)
    
    # Price gap persistence (gap fill patterns)
    price_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_fill = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1) + 1e-8)
    
    # Post-gap return patterns
    post_gap_returns = data['close'].pct_change(3).shift(-3)
    gap_impact = volume_gap.astype(float) * gap_fill * post_gap_returns
    
    # 5. Multi-Timeframe Trend Consistency
    # Short-term trend (3-day)
    def calc_slope(series, window):
        x = np.arange(window)
        slopes = series.rolling(window=window).apply(
            lambda y: stats.linregress(x, y[-window:])[0] if len(y[-window:]) == window else np.nan,
            raw=False
        )
        return slopes
    
    trend_3d = calc_slope(data['close'], 3)
    volume_trend_3d = calc_slope(data['volume'], 3)
    
    # Medium-term trend (10-day)
    trend_10d = calc_slope(data['close'], 10)
    volume_trend_10d = calc_slope(data['volume'], 10)
    
    # Trend alignment score
    direction_consistency = np.sign(trend_3d) * np.sign(trend_10d)
    magnitude_alignment = 1 - (abs(trend_3d - trend_10d) / (abs(trend_3d) + abs(trend_10d) + 1e-8))
    volume_confirmation = np.sign(volume_trend_3d) * np.sign(trend_3d) + np.sign(volume_trend_10d) * np.sign(trend_10d)
    
    trend_alignment = direction_consistency * magnitude_alignment * (1 + 0.2 * volume_confirmation)
    
    # Combine all factors with equal weights
    divergence_factor = pd.Series(divergence_signal, index=data.index).fillna(0)
    volatility_factor = pd.Series(regime_adjusted_returns, index=data.index).fillna(0)
    pressure_factor = pd.Series(pressure_accumulation, index=data.index).fillna(0)
    gap_factor = pd.Series(gap_impact, index=data.index).fillna(0)
    trend_factor = pd.Series(trend_alignment, index=data.index).fillna(0)
    
    # Normalize factors
    factors = [divergence_factor, volatility_factor, pressure_factor, gap_factor, trend_factor]
    normalized_factors = []
    
    for factor in factors:
        # Remove outliers and normalize
        factor_clean = factor.clip(lower=factor.quantile(0.05), upper=factor.quantile(0.95))
        normalized = (factor_clean - factor_clean.mean()) / factor_clean.std()
        normalized_factors.append(normalized)
    
    # Composite factor (equal weighted combination)
    composite_factor = sum(normalized_factors) / len(normalized_factors)
    
    return composite_factor
