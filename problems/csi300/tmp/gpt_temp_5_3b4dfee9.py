import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-scale Intraday Price Efficiency
    # Short-term Intraday Efficiency
    daily_range_efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    daily_range_efficiency = daily_range_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    intraday_momentum_divergence = (data['high'] / data['close']) - (data['low'] / data['close'])
    short_term_efficiency = daily_range_efficiency * intraday_momentum_divergence
    
    # Medium-term Efficiency Patterns
    # 5-day rolling correlation of daily range efficiencies
    efficiency_corr = daily_range_efficiency.rolling(window=5, min_periods=3).corr(daily_range_efficiency.shift(1))
    efficiency_corr = efficiency_corr.fillna(0)
    
    # Efficiency trend slope using 5-day linear regression
    def linear_trend(series):
        if len(series) < 3:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    efficiency_trend = daily_range_efficiency.rolling(window=5, min_periods=3).apply(linear_trend, raw=False)
    efficiency_trend = efficiency_trend.fillna(0)
    
    # Efficiency persistence
    efficiency_persistence = (daily_range_efficiency > daily_range_efficiency.rolling(window=5, min_periods=3).mean()).astype(int)
    efficiency_persistence = efficiency_persistence.rolling(window=3, min_periods=1).sum()
    
    # Multi-scale Efficiency Integration
    multi_scale_efficiency = short_term_efficiency * efficiency_persistence * (1 + efficiency_trend)
    
    # Volume-Price Divergence Analysis
    # Short-term Divergence Signals
    price_momentum = (data['close'] / data['close'].shift(5) - 1).fillna(0)
    volume_momentum = (data['volume'] / data['volume'].shift(5) - 1).fillna(0)
    price_volume_divergence = price_momentum - volume_momentum
    
    # Volume Confirmation Strength
    volume_concentration = data['volume'] * np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    volume_concentration = volume_concentration.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    volume_persistence = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    volume_persistence = volume_persistence.fillna(1)
    
    volume_weighted_momentum = data['volume'] * price_momentum
    
    # Divergence Pattern Assessment
    divergence_sign = np.sign(price_volume_divergence)
    divergence_persistence = (divergence_sign == divergence_sign.shift(1)).astype(int)
    divergence_persistence = divergence_persistence.rolling(window=3, min_periods=1).sum()
    
    avg_divergence_persistence = divergence_persistence.rolling(window=10, min_periods=5).mean().fillna(0)
    divergence_strength = divergence_persistence / (avg_divergence_persistence + 1e-8)
    
    # Volume validation of divergence patterns
    volume_divergence_confirmation = volume_concentration * divergence_strength
    
    # Volatility-Structure Enhanced Components
    # Multi-period Directional Volatility
    upside_volatility = (data['high'] - data['close']).rolling(window=3, min_periods=2).mean().fillna(0)
    downside_volatility = (data['close'] - data['low']).rolling(window=3, min_periods=2).mean().fillna(0)
    volatility_asymmetry = upside_volatility / (downside_volatility + 1e-8)
    volatility_asymmetry = volatility_asymmetry.replace([np.inf, -np.inf], 1).fillna(1)
    
    # True Range Volatility Analysis
    true_range = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': np.abs(data['high'] - data['close'].shift(1)),
        'lc': np.abs(data['low'] - data['close'].shift(1))
    }).max(axis=1).fillna(0)
    
    volatility_acceleration = true_range / (true_range.shift(5) + 1e-8)
    volatility_acceleration = volatility_acceleration.replace([np.inf, -np.inf], 1).fillna(1)
    
    avg_true_range = true_range.rolling(window=20, min_periods=10).mean().fillna(true_range.mean())
    volatility_regime = (true_range > avg_true_range).astype(int)
    
    # Volatility-Volume Integration
    volatility_volume_component = (intraday_momentum_divergence / (true_range + 1e-8)) * volume_concentration * volatility_asymmetry
    
    # Compression Breakout with Volume Validation
    # Price Compression Detection
    price_range_8d = (data['high'] - data['low']).rolling(window=8, min_periods=5).mean().fillna(0)
    price_range_20d = (data['high'] - data['low']).rolling(window=20, min_periods=10).mean().fillna(0)
    compression_intensity = (price_range_8d < 0.6 * price_range_20d).astype(float)
    
    # Volume-Validated Breakout Signals
    close_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    volume_expansion = data['volume'] / (data['volume'].shift(1) + 1e-8)
    breakout_strength = close_position * volume_expansion
    
    # Compression-Breakout Integration
    compression_breakout = compression_intensity * breakout_strength
    
    # Regime-Adaptive Signal Construction
    # Volatility Regime Classification
    high_vol_weight = volatility_regime
    low_vol_weight = 1 - volatility_regime
    
    # Regime-Specific Weighting
    regime_mean_reversion = multi_scale_efficiency * price_volume_divergence * high_vol_weight
    regime_momentum = multi_scale_efficiency * volume_weighted_momentum * low_vol_weight
    
    # Transition blending
    transition_weight = np.abs(volatility_acceleration - 1)
    regime_component = regime_mean_reversion * (1 - transition_weight) + regime_momentum * transition_weight
    
    # Momentum Regime Enhancement
    signal_strength = regime_component.rolling(window=3, min_periods=2).apply(linear_trend, raw=False).fillna(0)
    
    # Composite Factor Generation
    # Primary Signal Integration
    primary_signal = multi_scale_efficiency * price_volume_divergence
    volatility_adjusted = primary_signal * volatility_volume_component
    
    # Multi-scale Divergence Strength Assessment
    divergence_consistency = (price_volume_divergence * price_volume_divergence.shift(3)).fillna(0)
    volume_validation = volume_divergence_confirmation * np.abs(divergence_consistency)
    
    # Final Alpha Output
    alpha_factor = (
        regime_component * (1 + signal_strength) +
        volatility_adjusted * volume_validation +
        compression_breakout * divergence_strength
    )
    
    return alpha_factor.fillna(0)
