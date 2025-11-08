import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Acceleration-Weighted Volatility Asymmetry Factor
    Combines volatility asymmetry, momentum acceleration, and volume confirmation
    with regime-adaptive weighting for enhanced predictive power.
    """
    data = df.copy()
    
    # 1. Calculate Volatility Asymmetry with Acceleration
    # Directional Volatility Asymmetry
    upside_days = data['close'] > data['open']
    downside_days = data['close'] < data['open']
    
    # Calculate daily volatility (high-low range)
    daily_vol = data['high'] - data['low']
    
    # Upside volatility (only on up days)
    upside_vol = daily_vol.where(upside_days, 0)
    # Downside volatility (only on down days)
    downside_vol = daily_vol.where(downside_days, 0)
    
    # Rolling 5-day volatility asymmetry ratio
    upside_vol_5d = upside_vol.rolling(window=5, min_periods=3).mean()
    downside_vol_5d = downside_vol.rolling(window=5, min_periods=3).mean()
    
    # Avoid division by zero
    volatility_asymmetry = upside_vol_5d / (downside_vol_5d + 1e-8)
    
    # Volatility Acceleration
    vol_asymmetry_roc = volatility_asymmetry.pct_change(periods=1)
    vol_acceleration = vol_asymmetry_roc.diff()
    
    # Acceleration consistency over 5 days
    accel_consistency = vol_acceleration.rolling(window=5, min_periods=3).std()
    
    # 2. Analyze Volume-Momentum Phase Relationship
    # Price Momentum Characteristics
    momentum_5d = data['close'].pct_change(periods=5)
    momentum_acceleration = momentum_5d.diff()
    
    # Volume Confirmation Patterns
    volume_slope = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    
    # Volume-Momentum correlation (5-day rolling)
    def rolling_corr(x):
        prices = x[:5]  # First 5 elements are prices
        volumes = x[5:]  # Last 5 elements are volumes
        if len(prices) >= 3 and len(volumes) >= 3:
            return np.corrcoef(prices, volumes)[0, 1] if not (np.std(prices) == 0 or np.std(volumes) == 0) else 0
        return 0
    
    # Create combined series for correlation calculation
    price_changes = data['close'].pct_change().fillna(0)
    volume_norm = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    
    combined_data = pd.concat([price_changes, volume_norm], axis=1)
    volume_momentum_corr = combined_data.rolling(window=5, min_periods=3).apply(
        lambda x: rolling_corr(x.values.flatten()), raw=False
    ).iloc[:, 0]
    
    # Detect Constructive/Destructive Phases
    momentum_direction = np.sign(momentum_5d)
    vol_asymmetry_direction = np.sign(volatility_asymmetry - 1)  # >1 means upside bias
    
    phase_alignment = (momentum_direction == vol_asymmetry_direction).astype(int)
    phase_strength = (abs(momentum_5d) * abs(volatility_asymmetry - 1)).fillna(0)
    
    # 3. Generate Regime-Adaptive Composite Signal
    # Market Regime Classification
    daily_ranges = data['high'] - data['low']
    range_percentile = daily_ranges.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Define volatility regimes
    low_vol_regime = (range_percentile <= 0.3).astype(int)
    medium_vol_regime = ((range_percentile > 0.3) & (range_percentile <= 0.7)).astype(int)
    high_vol_regime = (range_percentile > 0.7).astype(int)
    
    # Combine Asymmetry and Momentum Components
    asymmetry_momentum_component = volatility_asymmetry * momentum_acceleration
    
    # Weight by volume confirmation
    volume_confirmation = volume_slope * volume_momentum_corr
    
    # Scale by phase relationship
    phase_adjusted = asymmetry_momentum_component * (1 + phase_alignment * phase_strength)
    
    # Apply Regime-Specific Adjustments
    # Different weighting schemes per volatility regime
    low_vol_weight = 1.2  # More sensitive in low volatility
    medium_vol_weight = 1.0  # Neutral weighting
    high_vol_weight = 0.8  # Less sensitive in high volatility
    
    regime_adjusted_signal = (
        phase_adjusted * volume_confirmation * low_vol_regime * low_vol_weight +
        phase_adjusted * volume_confirmation * medium_vol_regime * medium_vol_weight +
        phase_adjusted * volume_confirmation * high_vol_regime * high_vol_weight
    )
    
    # Incorporate regime transition smoothing
    regime_persistence = (
        low_vol_regime.rolling(window=3, min_periods=1).mean() +
        medium_vol_regime.rolling(window=3, min_periods=1).mean() +
        high_vol_regime.rolling(window=3, min_periods=1).mean()
    )
    
    # Final composite signal with smoothing
    final_signal = regime_adjusted_signal * (1 + 0.1 * regime_persistence)
    
    # Normalize the final signal
    signal_zscore = (final_signal - final_signal.rolling(window=20, min_periods=10).mean()) / (final_signal.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return signal_zscore
