import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Asymmetric Volatility-Efficiency Framework
    # Volatility Asymmetry
    upside_vol = df['high'] - df['close']
    downside_vol = df['close'] - df['low']
    upside_vol = np.maximum(upside_vol, 0)
    downside_vol = np.maximum(downside_vol, 0)
    
    upside_vol_avg = upside_vol.rolling(window=10, min_periods=5).mean()
    downside_vol_avg = downside_vol.rolling(window=10, min_periods=5).mean()
    volatility_asymmetry = upside_vol_avg / downside_vol_avg
    
    # Price-Volume Efficiency
    intraday_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    volume_percentile = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    efficiency_signal = intraday_efficiency * volume_percentile
    
    # Core Asymmetric Signal
    efficiency_momentum = efficiency_signal - efficiency_signal.shift(3)
    core_asymmetric_signal = volatility_asymmetry * efficiency_signal * np.sign(efficiency_momentum)
    
    # Momentum-Pressure Confirmation System
    # Quality Momentum
    returns = df['close'].pct_change()
    directional_sign = np.sign(returns)
    consecutive_same_sign = directional_sign.rolling(window=5, min_periods=3).apply(
        lambda x: len(set(x)) == 1 if len(x) == 5 else np.nan, raw=False
    ).astype(float)
    
    daily_range = df['high'] - df['low']
    avg_daily_range_5d = daily_range.rolling(window=5, min_periods=3).mean()
    return_5d = df['close'].pct_change(5)
    return_to_volatility = return_5d / avg_daily_range_5d
    quality_momentum = consecutive_same_sign * return_to_volatility
    
    # Microstructure Pressure
    price_rejection = (df['high'] - df['close']) / (df['close'] - df['low'])
    price_rejection = price_rejection.replace([np.inf, -np.inf], np.nan)
    volume_concentration = (df['high'] - df['low']) / df['volume']
    pressure_score = price_rejection * volume_concentration
    
    # Enhanced Core Signal
    enhanced_core_signal = core_asymmetric_signal * quality_momentum * pressure_score
    
    # Regime-Adaptive Processing
    # Volatility Regime
    recent_avg_range = daily_range.rolling(window=5, min_periods=3).mean()
    historical_avg_range = daily_range.rolling(window=20, min_periods=10).mean()
    volatility_regime = (recent_avg_range > historical_avg_range).astype(float)
    
    # Component Weight
    def rolling_corr(x, y, window):
        return pd.Series(x).rolling(window=window).corr(pd.Series(y))
    
    weight1 = np.abs(rolling_corr(efficiency_signal, quality_momentum, 5))
    weight2 = np.abs(rolling_corr(volatility_asymmetry, pressure_score, 5))
    component_weight = weight1 * weight2
    
    # Regime-Adapted Signal
    regime_adapted_signal = enhanced_core_signal * component_weight * volatility_regime
    
    # Final Alpha Factor
    avg_daily_range_10d = daily_range.rolling(window=10, min_periods=5).mean()
    base_factor = regime_adapted_signal / avg_daily_range_10d
    
    # Adjustment
    volatility_asymmetry_trend = volatility_asymmetry.diff(5).abs()
    final_factor = base_factor * np.sign(efficiency_momentum) * volatility_asymmetry_trend
    
    return final_factor
