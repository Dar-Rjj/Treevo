import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-scaled momentum acceleration,
    volume-price divergence confirmation, intraday pattern regime detection,
    and multi-timeframe geometric integration.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for volatility calculations
    returns = df['close'].pct_change()
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Volatility-Scaled Momentum Acceleration
        # Multi-Timeframe Momentum
        mom_3 = (df['close'].iloc[i] / df['close'].iloc[i-3] - 1) if i >= 3 else 0
        mom_5 = (df['close'].iloc[i] / df['close'].iloc[i-5] - 1) if i >= 5 else 0
        mom_10 = (df['close'].iloc[i] / df['close'].iloc[i-10] - 1) if i >= 10 else 0
        
        # Momentum Acceleration
        accel_3_5 = mom_5 - mom_3
        accel_5_10 = mom_10 - mom_5
        accel_persistence = accel_3_5 - accel_5_10
        
        # Volatility Scaling
        vol_3 = returns.iloc[max(0, i-2):i+1].std() if i >= 2 else 1
        vol_5 = returns.iloc[max(0, i-4):i+1].std() if i >= 4 else 1
        vol_10 = returns.iloc[max(0, i-9):i+1].std() if i >= 9 else 1
        
        # Combined Signal
        scaled_3_5 = accel_3_5 / vol_3 if vol_3 != 0 else 0
        scaled_5_10 = accel_5_10 / vol_5 if vol_5 != 0 else 0
        scaled_persistence = accel_persistence / vol_10 if vol_10 != 0 else 0
        
        # Avoid zeros in geometric mean
        momentum_component = np.cbrt(
            np.sign(scaled_3_5) * np.abs(scaled_3_5 + 1e-8) *
            np.sign(scaled_5_10) * np.abs(scaled_5_10 + 1e-8) *
            np.sign(scaled_persistence) * np.abs(scaled_persistence + 1e-8)
        )
        
        # 2. Volume-Price Divergence Confirmation
        # Price Strength Components
        high_low_range = df['high'].iloc[i] - df['low'].iloc[i]
        if high_low_range == 0:
            intraday_eff = 0
            high_low_dom = 0
            price_persistence = 0
        else:
            intraday_eff = (df['close'].iloc[i] - df['open'].iloc[i]) / high_low_range
            high_low_dom = (df['close'].iloc[i] - df['low'].iloc[i]) / high_low_range
            price_persistence = (df['close'].iloc[i] - df['close'].iloc[i-1]) / high_low_range if i >= 1 else 0
        
        # Volume Divergence Signals
        if i >= 3:
            vol_ma_3 = (df['volume'].iloc[i-1] + df['volume'].iloc[i-2] + df['volume'].iloc[i-3]) / 3
            volume_ratio = df['volume'].iloc[i] / vol_ma_3 if vol_ma_3 != 0 else 1
        else:
            volume_ratio = 1
            
        volume_trend = df['volume'].iloc[i] / df['volume'].iloc[i-1] if i >= 1 and df['volume'].iloc[i-1] != 0 else 1
        
        if i >= 6:
            vol_ma_6 = (df['volume'].iloc[i-4] + df['volume'].iloc[i-5] + df['volume'].iloc[i-6]) / 3
            volume_stability = df['volume'].iloc[i] / vol_ma_6 if vol_ma_6 != 0 else 1
        else:
            volume_stability = 1
        
        # Volume-Price Alignment
        strong_price_high_vol = price_persistence * volume_ratio
        weak_price_high_vol = -price_persistence * volume_ratio
        price_efficiency_vol_trend = intraday_eff * volume_trend
        
        # Combined Signal
        core_divergence = (strong_price_high_vol + weak_price_high_vol + price_efficiency_vol_trend) / 3
        volume_component = core_divergence * volume_stability
        
        # 3. Intraday Pattern Regime Detection
        # Morning Session Analysis
        opening_gap_strength = (df['open'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1] if i >= 1 else 0
        
        if high_low_range == 0:
            morning_range_eff = 0
            morning_support_pressure = 0
            afternoon_momentum = 0
            afternoon_support = 0
            closing_efficiency = 0
        else:
            morning_range_eff = (df['high'].iloc[i] - df['open'].iloc[i]) / high_low_range
            morning_support_pressure = (df['open'].iloc[i] - df['low'].iloc[i]) / high_low_range
            afternoon_momentum = (df['close'].iloc[i] - df['high'].iloc[i]) / high_low_range
            afternoon_support = (df['close'].iloc[i] - df['low'].iloc[i]) / high_low_range
            closing_efficiency = (df['close'].iloc[i] - df['open'].iloc[i]) / high_low_range
        
        # Intraday Regime Classification and Weighted Strength
        if abs(morning_range_eff - morning_support_pressure) < 0.1:
            # Balanced regime
            regime_strength = (morning_range_eff + afternoon_momentum) / 2 * closing_efficiency
        elif morning_range_eff > morning_support_pressure:
            # Morning-driven regime
            regime_strength = morning_range_eff * closing_efficiency
        else:
            # Afternoon-driven regime
            regime_strength = afternoon_momentum * closing_efficiency
        
        intraday_component = regime_strength * (1 + opening_gap_strength)
        
        # 4. Multi-Timeframe Geometric Integration
        # Short-term Components (1-3 days)
        price_momentum = df['close'].iloc[i] / df['close'].iloc[i-2] - 1 if i >= 2 else 0
        volume_acceleration = df['volume'].iloc[i] / df['volume'].iloc[i-2] if i >= 2 and df['volume'].iloc[i-2] != 0 else 1
        range_utilization = (df['close'].iloc[i] - df['close'].iloc[i-1]) / high_low_range if i >= 1 and high_low_range != 0 else 0
        
        # Medium-term Components (5-10 days)
        price_trend = df['close'].iloc[i] / df['close'].iloc[i-7] - 1 if i >= 7 else 0
        volume_trend_mtf = df['volume'].iloc[i] / df['volume'].iloc[i-7] if i >= 7 and df['volume'].iloc[i-7] != 0 else 1
        
        if i >= 7:
            avg_range_7d = np.mean([df['high'].iloc[j] - df['low'].iloc[j] for j in range(i-6, i+1)])
            volatility_efficiency = high_low_range / avg_range_7d if avg_range_7d != 0 else 1
        else:
            volatility_efficiency = 1
        
        # Signal Geometric Combination
        short_term_geo = np.cbrt(
            np.sign(price_momentum) * np.abs(price_momentum + 1e-8) *
            np.sign(volume_acceleration) * np.abs(volume_acceleration + 1e-8) *
            np.sign(range_utilization) * np.abs(range_utilization + 1e-8)
        )
        
        medium_term_geo = np.cbrt(
            np.sign(price_trend) * np.abs(price_trend + 1e-8) *
            np.sign(volume_trend_mtf) * np.abs(volume_trend_mtf + 1e-8) *
            np.sign(volatility_efficiency) * np.abs(volatility_efficiency + 1e-8)
        )
        
        multi_timeframe_component = short_term_geo * medium_term_geo
        
        # 5. Final Alpha Factor - Geometric Mean of all four components
        components = [
            np.sign(momentum_component) * np.abs(momentum_component + 1e-8),
            np.sign(volume_component) * np.abs(volume_component + 1e-8),
            np.sign(intraday_component) * np.abs(intraday_component + 1e-8),
            np.sign(multi_timeframe_component) * np.abs(multi_timeframe_component + 1e-8)
        ]
        
        # Geometric mean preserving sign
        result.iloc[i] = np.prod([np.sign(c) for c in components]) * np.power(np.prod([np.abs(c) for c in components]), 1/4)
    
    return result
