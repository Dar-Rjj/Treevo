import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum Quality Factor
    Combines multi-timeframe momentum quality with volume confirmation and volatility regime adaptation
    """
    
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Quality Assessment
    # Short-term momentum (1-3 days)
    mom_1d = data['close'].pct_change(1)
    mom_2d = data['close'].pct_change(2)
    mom_3d = data['close'].pct_change(3)
    
    # Medium-term momentum (5-10 days)
    mom_5d = data['close'].pct_change(5)
    mom_10d = data['close'].pct_change(10)
    
    # Calculate momentum slopes using linear regression coefficients
    def rolling_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) == window and not np.all(y == y[0]):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.iloc[i] = slope
        return slopes
    
    mom_slope_5d = rolling_slope(data['close'], 5)
    mom_slope_10d = rolling_slope(data['close'], 10)
    
    # Momentum Acceleration Signals
    accel_short = mom_3d - mom_1d  # 3-day vs 1-day momentum
    accel_medium = mom_10d - mom_5d  # 10-day vs 5-day momentum
    accel_cross = mom_slope_10d - mom_slope_5d  # Slope acceleration
    
    # Momentum Consistency Score
    def calculate_consistency_score(mom_signals):
        """Calculate consistency across multiple momentum signals"""
        signs = pd.DataFrame({f'signal_{i}': np.sign(sig) for i, sig in enumerate(mom_signals)})
        consistency = signs.apply(lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 0, axis=1)
        return consistency
    
    momentum_signals = [mom_1d, mom_3d, mom_5d, mom_10d]
    consistency_score = calculate_consistency_score(momentum_signals)
    
    # Magnitude progression quality
    mom_magnitudes = [abs(mom_1d), abs(mom_3d), abs(mom_5d), abs(mom_10d)]
    magnitude_trend = pd.DataFrame(mom_magnitudes).T.rolling(3, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    ).mean(axis=1)
    
    # Dynamic Volatility Regime Classification
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Rolling volatility measures
    vol_5d = true_range.rolling(5).mean()
    vol_20d = true_range.rolling(20).mean()
    vol_ratio = vol_5d / vol_20d
    
    # Price Range Efficiency
    daily_range = data['high'] - data['low']
    range_efficiency = daily_range / true_range.rolling(10).mean()
    
    # Volatility regime classification
    def classify_regime(vol_ratio, range_eff):
        high_vol = (vol_ratio > 1.2) & (range_eff > 1.1)
        low_vol = (vol_ratio < 0.8) & (range_eff < 0.9)
        transition = ~high_vol & ~low_vol
        return high_vol.astype(int), low_vol.astype(int), transition.astype(int)
    
    high_vol_regime, low_vol_regime, transition_regime = classify_regime(vol_ratio, range_efficiency)
    
    # Volume-Price Quality Confirmation
    # Volume trend and momentum
    vol_ma_5 = data['volume'].rolling(5).mean()
    vol_ma_20 = data['volume'].rolling(20).mean()
    vol_momentum = vol_ma_5 / vol_ma_20
    
    # Volume concentration (recent volume vs historical)
    vol_concentration = data['volume'] / data['volume'].rolling(20).mean()
    
    # Volume during momentum quality changes
    vol_quality = vol_momentum * np.sign(accel_short.fillna(0))
    
    # Volume-volatility relationship
    vol_vol_ratio = vol_concentration / (vol_ratio.replace(0, 0.001))
    
    # Generate Adaptive Quality Factor
    # Regime-adaptive parameters
    # High volatility: shorter lookbacks, quality focus
    high_vol_factor = (
        (mom_3d.fillna(0) * 0.6 + mom_5d.fillna(0) * 0.4) * 
        consistency_score.fillna(0) * 
        (1 + vol_quality.fillna(0))
    )
    
    # Low volatility: longer trends, consistency emphasis
    low_vol_factor = (
        (mom_10d.fillna(0) * 0.7 + mom_slope_10d.fillna(0) * 0.3) * 
        magnitude_trend.fillna(0) * 
        (1 + vol_concentration.fillna(0))
    )
    
    # Transition regimes: balanced approach
    transition_factor = (
        (mom_5d.fillna(0) * 0.5 + mom_10d.fillna(0) * 0.3 + accel_cross.fillna(0) * 0.2) * 
        (consistency_score.fillna(0) * 0.6 + magnitude_trend.fillna(0) * 0.4) * 
        (1 + vol_vol_ratio.fillna(0) * 0.5)
    )
    
    # Final factor construction with regime adaptation
    final_factor = (
        high_vol_regime * high_vol_factor +
        low_vol_regime * low_vol_factor +
        transition_regime * transition_factor
    )
    
    # Scale by volatility regime strength
    regime_strength = abs(vol_ratio - 1)
    scaled_factor = final_factor * (1 + regime_strength.fillna(0))
    
    # Normalize and clean
    factor_series = scaled_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor_series
