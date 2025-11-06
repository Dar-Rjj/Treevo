import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volume Efficiency Divergence with Regime Adaptation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(max(60, len(df))):  # Ensure enough data for calculations
        if i < 60:  # Need sufficient history for meaningful calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-Timeframe Volume Efficiency Calculation
        # Short-term (3-day) efficiency
        short_window = 3
        if i >= short_window:
            short_returns = (current_data['close'].iloc[i] / current_data['close'].iloc[i-short_window] - 1)
            short_volume = current_data['volume'].iloc[i-short_window:i+1].sum()
            short_efficiency = short_returns / short_volume if short_volume > 0 else 0
            
            # Volume concentration during price moves
            short_price_range = (current_data['high'].iloc[i-short_window:i+1].max() - 
                               current_data['low'].iloc[i-short_window:i+1].min())
            short_volume_concentration = short_volume / short_price_range if short_price_range > 0 else 0
        else:
            short_efficiency = 0
            short_volume_concentration = 0
        
        # Medium-term (10-day) efficiency
        medium_window = 10
        if i >= medium_window:
            medium_returns = (current_data['close'].iloc[i] / current_data['close'].iloc[i-medium_window] - 1)
            medium_volume = current_data['volume'].iloc[i-medium_window:i+1].sum()
            medium_efficiency = medium_returns / medium_volume if medium_volume > 0 else 0
            
            # Price range utilization efficiency
            medium_price_range = (current_data['high'].iloc[i-medium_window:i+1].max() - 
                                current_data['low'].iloc[i-medium_window:i+1].min())
            medium_range_efficiency = medium_returns / medium_price_range if medium_price_range > 0 else 0
        else:
            medium_efficiency = 0
            medium_range_efficiency = 0
        
        # Long-term (20-day) efficiency
        long_window = 20
        if i >= long_window:
            long_returns = (current_data['close'].iloc[i] / current_data['close'].iloc[i-long_window] - 1)
            long_volume = current_data['volume'].iloc[i-long_window:i+1].sum()
            long_efficiency = long_returns / long_volume if long_volume > 0 else 0
            
            # Efficiency trend acceleration
            if i >= long_window * 2:
                prev_long_efficiency = (current_data['close'].iloc[i-long_window] / 
                                      current_data['close'].iloc[i-long_window*2] - 1) / long_volume
                efficiency_acceleration = long_efficiency - prev_long_efficiency
            else:
                efficiency_acceleration = 0
        else:
            long_efficiency = 0
            efficiency_acceleration = 0
        
        # Efficiency Quality Assessment
        # Efficiency persistence (consecutive days with positive efficiency divergence)
        efficiency_persistence = 0
        if i >= 5:
            recent_efficiencies = []
            for j in range(5):
                if i-j >= medium_window:
                    ret = (current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-medium_window] - 1)
                    vol = current_data['volume'].iloc[i-j-medium_window:i-j+1].sum()
                    eff = ret / vol if vol > 0 else 0
                    recent_efficiencies.append(eff)
            
            if len(recent_efficiencies) >= 3:
                # Count consecutive increases in efficiency
                for k in range(1, len(recent_efficiencies)):
                    if recent_efficiencies[k] > recent_efficiencies[k-1]:
                        efficiency_persistence += 1
        
        # Regime-Adaptive Dynamics
        # Volatility regime classification
        if i >= 20:
            # 5-day vs 20-day efficiency volatility
            recent_efficiencies_5d = []
            recent_efficiencies_20d = []
            
            for j in range(5):
                if i-j >= 5:
                    ret = (current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-5] - 1)
                    vol = current_data['volume'].iloc[i-j-5:i-j+1].sum()
                    eff = ret / vol if vol > 0 else 0
                    recent_efficiencies_5d.append(eff)
            
            for j in range(20):
                if i-j >= 20:
                    ret = (current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-20] - 1)
                    vol = current_data['volume'].iloc[i-j-20:i-j+1].sum()
                    eff = ret / vol if vol > 0 else 0
                    recent_efficiencies_20d.append(eff)
            
            if len(recent_efficiencies_5d) >= 3 and len(recent_efficiencies_20d) >= 10:
                vol_5d = np.std(recent_efficiencies_5d) if len(recent_efficiencies_5d) > 1 else 0
                vol_20d = np.std(recent_efficiencies_20d) if len(recent_efficiencies_20d) > 1 else 0
                volatility_regime = vol_5d / vol_20d if vol_20d > 0 else 1
            else:
                volatility_regime = 1
        else:
            volatility_regime = 1
        
        # Advanced Breakout Analytics
        breakout_efficiency = 0
        if i >= 20:
            # Volume surge detection (>2x 20-day average)
            avg_volume_20d = current_data['volume'].iloc[i-20:i+1].mean()
            current_volume = current_data['volume'].iloc[i]
            volume_surge = current_volume > (2 * avg_volume_20d) if avg_volume_20d > 0 else False
            
            if volume_surge:
                # Breakout efficiency assessment
                price_change = (current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1)
                breakout_efficiency = price_change / current_volume if current_volume > 0 else 0
        
        # Risk & Momentum Assessment
        risk_adjusted_efficiency = 0
        if i >= 20:
            # Efficiency per unit volatility
            recent_returns = current_data['close'].iloc[i-19:i+1].pct_change().dropna()
            if len(recent_returns) > 1:
                return_volatility = recent_returns.std()
                if return_volatility > 0:
                    risk_adjusted_efficiency = medium_efficiency / return_volatility
        
        # Final Factor Construction
        # Core efficiency factor with regime adjustment
        core_efficiency = (short_efficiency * 0.3 + 
                          medium_efficiency * 0.4 + 
                          long_efficiency * 0.3)
        
        # Multi-timeframe convergence
        timeframe_convergence = 0
        if (short_efficiency > 0 and medium_efficiency > 0 and long_efficiency > 0):
            timeframe_convergence = 1
        elif (short_efficiency < 0 and medium_efficiency < 0 and long_efficiency < 0):
            timeframe_convergence = -1
        
        # Regime-aware adjustment
        regime_adjustment = 1.0
        if volatility_regime > 1.5:  # High volatility regime
            regime_adjustment = 0.7  # Reduce signal strength
        elif volatility_regime < 0.7:  # Low volatility regime
            regime_adjustment = 1.3  # Enhance signal strength
        
        # Breakout enhancement
        breakout_multiplier = 1.0 + (breakout_efficiency * 2.0 if breakout_efficiency > 0 else 0)
        
        # Combine all components
        final_factor = (core_efficiency * regime_adjustment * 
                       (1 + timeframe_convergence * 0.2) * 
                       breakout_multiplier * 
                       (1 + efficiency_persistence * 0.1) * 
                       (1 + risk_adjusted_efficiency * 0.5))
        
        result.iloc[i] = final_factor
    
    return result
