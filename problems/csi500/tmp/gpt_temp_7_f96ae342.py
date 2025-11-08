import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multiple market dynamics:
    - Volatility-adjusted multi-timeframe momentum
    - Multi-timeframe price-volume efficiency
    - Range expansion with momentum confirmation
    - Volatility-regime adaptive trend strength
    - Session-based pattern recognition
    """
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Required lookback periods
    max_lookback = max(30, 20, 15, 14, 9, 6, 5, 4, 3, 2)
    
    for i in range(max_lookback, len(data)):
        current_data = data.iloc[:i+1]
        
        # 1. Volatility-Adjusted Multi-Timeframe Momentum
        momentum_2d = current_data['close'].iloc[i] / current_data['close'].iloc[i-2] - 1 if i >= 2 else 0
        momentum_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1 if i >= 5 else 0
        momentum_10d = current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1 if i >= 10 else 0
        momentum_20d = current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1 if i >= 20 else 0
        
        # Calculate volatilities
        def calc_volatility(data, period):
            if len(data) < period + 1:
                return 1.0  # Avoid division by zero
            returns = data['close'].pct_change().iloc[-period:]
            return returns.std() if len(returns) > 1 else 1.0
        
        vol_2d = calc_volatility(current_data.iloc[i-1:i+1], 2) if i >= 2 else 1.0
        vol_5d = calc_volatility(current_data.iloc[i-4:i+1], 5) if i >= 5 else 1.0
        vol_10d = calc_volatility(current_data.iloc[i-9:i+1], 10) if i >= 10 else 1.0
        vol_20d = calc_volatility(current_data.iloc[i-19:i+1], 20) if i >= 20 else 1.0
        
        # Volatility-scaled momentum
        vol_scaled_momentum = (
            (momentum_2d / (vol_2d + 1e-8) if vol_2d > 0 else 0) +
            (momentum_5d / (vol_5d + 1e-8) if vol_5d > 0 else 0) +
            (momentum_10d / (vol_10d + 1e-8) if vol_10d > 0 else 0) +
            (momentum_20d / (vol_20d + 1e-8) if vol_20d > 0 else 0)
        ) / 4
        
        # Momentum convergence
        momentum_convergence = (
            np.sign(momentum_2d) * np.sign(momentum_5d) * 
            np.sign(momentum_10d) * np.sign(momentum_20d) * 
            vol_scaled_momentum
        )
        
        # 2. Multi-Timeframe Price-Volume Efficiency
        # Price efficiency components
        intraday_eff = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (
            current_data['high'].iloc[i] - current_data['low'].iloc[i] + 1e-8
        ) if i >= 0 else 0
        
        eff_3d = (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / (
            current_data['high'].iloc[i-2:i+1].max() - current_data['low'].iloc[i-2:i+1].min() + 1e-8
        ) if i >= 3 else 0
        
        eff_7d = (current_data['close'].iloc[i] - current_data['close'].iloc[i-7]) / (
            current_data['high'].iloc[i-6:i+1].max() - current_data['low'].iloc[i-6:i+1].min() + 1e-8
        ) if i >= 7 else 0
        
        eff_15d = (current_data['close'].iloc[i] - current_data['close'].iloc[i-15]) / (
            current_data['high'].iloc[i-14:i+1].max() - current_data['low'].iloc[i-14:i+1].min() + 1e-8
        ) if i >= 15 else 0
        
        # Volume confirmation
        def volume_ratio(data, period):
            if len(data) < period + 1:
                return 1.0
            current_vol = data['volume'].iloc[i]
            avg_vol = data['volume'].iloc[i-period:i+1].mean()
            return current_vol / (avg_vol + 1e-8)
        
        vol_ratio_intra = volume_ratio(current_data, 4) if i >= 4 else 1.0
        vol_ratio_3d = volume_ratio(current_data, 2) if i >= 2 else 1.0
        vol_ratio_7d = volume_ratio(current_data, 6) if i >= 6 else 1.0
        vol_ratio_15d = volume_ratio(current_data, 14) if i >= 14 else 1.0
        
        # Combined efficiency signal
        efficiency_convergence = (intraday_eff + eff_3d + eff_7d + eff_15d) / 4
        volume_aligned_eff = efficiency_convergence * (
            vol_ratio_intra * vol_ratio_3d * vol_ratio_7d * vol_ratio_15d
        ) ** 0.25
        
        # 3. Range Expansion with Momentum Confirmation
        current_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / (current_data['close'].iloc[i] + 1e-8)
        
        def avg_range(data, period):
            if len(data) < period + 1:
                return current_range
            ranges = [
                (data['high'].iloc[j] - data['low'].iloc[j]) / (data['close'].iloc[j] + 1e-8)
                for j in range(i-period+1, i+1)
            ]
            return np.mean(ranges)
        
        avg_range_3d = avg_range(current_data, 3) if i >= 2 else current_range
        avg_range_7d = avg_range(current_data, 7) if i >= 6 else current_range
        avg_range_15d = avg_range(current_data, 15) if i >= 14 else current_range
        
        range_expansion_score = (
            (current_range / (avg_range_3d + 1e-8)) *
            (current_range / (avg_range_7d + 1e-8)) *
            (current_range / (avg_range_15d + 1e-8))
        )
        
        momentum_confirmed_expansion = range_expansion_score * (
            momentum_2d + momentum_5d + momentum_10d + momentum_20d
        ) / 4
        
        # 4. Volatility-Regime Adaptive Trend Strength
        trend_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1 if i >= 3 else 0
        trend_7d = current_data['close'].iloc[i] / current_data['close'].iloc[i-7] - 1 if i >= 7 else 0
        trend_15d = current_data['close'].iloc[i] / current_data['close'].iloc[i-15] - 1 if i >= 15 else 0
        trend_30d = current_data['close'].iloc[i] / current_data['close'].iloc[i-30] - 1 if i >= 30 else 0
        
        short_term_vol = calc_volatility(current_data.iloc[i-4:i+1], 5) if i >= 5 else 1.0
        medium_term_vol = calc_volatility(current_data.iloc[i-9:i+1], 10) if i >= 10 else 1.0
        long_term_vol = calc_volatility(current_data.iloc[i-29:i+1], 30) if i >= 30 else 1.0
        
        trend_convergence = (
            (trend_3d / (short_term_vol + 1e-8) if short_term_vol > 0 else 0) +
            (trend_7d / (short_term_vol + 1e-8) if short_term_vol > 0 else 0) +
            (trend_15d / (medium_term_vol + 1e-8) if medium_term_vol > 0 else 0) +
            (trend_30d / (long_term_vol + 1e-8) if long_term_vol > 0 else 0)
        ) / 4
        
        volatility_weighted_trend = trend_convergence * (
            abs(trend_3d * trend_7d * trend_15d * trend_30d) ** 0.25 * 
            np.sign(trend_3d + trend_7d + trend_15d + trend_30d)
        )
        
        # 5. Session-Based Pattern Recognition (simplified)
        gap_magnitude = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / (current_data['close'].iloc[i-1] + 1e-8) if i >= 1 else 0
        
        morning_momentum = (
            (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / 
            (current_data['open'].iloc[i] - current_data['low'].iloc[i] + 1e-8)
        ) if current_data['open'].iloc[i] > current_data['low'].iloc[i] else 1.0
        
        # Simplified session patterns (using daily data as proxy)
        midday_price = (current_data['high'].iloc[i] + current_data['low'].iloc[i]) / 2
        afternoon_strength = (current_data['close'].iloc[i] - midday_price) / (midday_price + 1e-8)
        
        closing_efficiency = (
            (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / 
            (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 1e-8)
        )
        
        session_pattern_strength = gap_magnitude * morning_momentum * afternoon_strength * closing_efficiency
        
        # Volume ratios as proxy for session volume patterns
        opening_vol_intensity = vol_ratio_intra
        closing_vol_confirmation = vol_ratio_intra  # Using same as proxy
        
        volume_confirmed_session = session_pattern_strength * opening_vol_intensity * closing_vol_confirmation
        
        # Final alpha combination (equal weighting for demonstration)
        final_alpha = (
            momentum_convergence +
            volume_aligned_eff +
            momentum_confirmed_expansion +
            volatility_weighted_trend +
            volume_confirmed_session
        ) / 5
        
        alpha.iloc[i] = final_alpha
    
    # Forward fill any NaN values
    alpha = alpha.fillna(method='ffill').fillna(0)
    
    return alpha
