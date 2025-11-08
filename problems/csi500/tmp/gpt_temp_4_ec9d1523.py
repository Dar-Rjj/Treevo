import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Need at least 20 days for calculations
            factor.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1].copy()
        
        # 1. Asymmetric Volatility Momentum with Volume Asymmetry
        # Volatility Asymmetry Calculation
        recent_10 = current_data.iloc[-10:]
        positive_returns = recent_10[recent_10['returns'] > 0]['returns']
        negative_returns = recent_10[recent_10['returns'] < 0]['returns']
        
        upside_vol = positive_returns.std() if len(positive_returns) > 1 else 1
        downside_vol = negative_returns.std() if len(negative_returns) > 1 else 1
        volatility_asymmetry = upside_vol / downside_vol if downside_vol != 0 else 1
        
        # Volume Asymmetry Pattern
        up_day_volume = recent_10[recent_10['returns'] > 0]['volume'].mean()
        down_day_volume = recent_10[recent_10['returns'] < 0]['volume'].mean()
        volume_asymmetry = up_day_volume / down_day_volume if down_day_volume != 0 else 1
        
        # Combined Asymmetry Signal
        base_signal_1 = volatility_asymmetry * volume_asymmetry
        momentum_enhancement = (current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1) if i >= 5 else 0
        factor_1 = base_signal_1 * momentum_enhancement
        
        # 2. Price-Volume Efficiency with Regime Detection
        # Efficiency Calculation
        if i >= 10:
            price_movement_efficiency = (current_data['close'].iloc[-1] - current_data['close'].iloc[-11]) / \
                                      sum(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) 
                                          for j in range(i-9, i+1))
            volume_efficiency = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-10:].mean()
            combined_efficiency = price_movement_efficiency * volume_efficiency
        else:
            combined_efficiency = 0
        
        # Market Regime Detection
        volatility_10d = current_data['returns'].iloc[-10:].std()
        volatility_50d = current_data['returns'].iloc[-50:].std() if i >= 50 else volatility_10d
        volatility_regime = 1 if volatility_10d > volatility_50d else -1
        
        trend_regime = 1 if (current_data['close'].iloc[-1] / current_data['close'].iloc[-21] - 1) > 0 else -1
        
        regime_multiplier = volatility_regime * trend_regime
        
        # Regime-Adjusted Efficiency
        base_factor_2 = combined_efficiency * regime_multiplier
        recent_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-4] - 1) if i >= 3 else 0
        factor_2 = base_factor_2 * recent_momentum
        
        # 3. Opening Gap Momentum with Intraday Confirmation
        # Gap Momentum Structure
        opening_gap = (current_data['open'].iloc[-1] / current_data['close'].iloc[-2] - 1) if i >= 1 else 0
        high_low_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        intraday_momentum = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / high_low_range if high_low_range != 0 else 0
        gap_momentum_interaction = opening_gap * intraday_momentum
        
        # Volume Confirmation Pattern
        gap_day_volume_ratio = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-2] if i >= 1 else 1
        pre_gap_volume_trend = (current_data['volume'].iloc[-2] / current_data['volume'].iloc[-4] - 1) if i >= 3 else 0
        volume_confirmation = gap_day_volume_ratio * pre_gap_volume_trend
        
        # Enhanced Gap Signal
        base_signal_3 = gap_momentum_interaction * volume_confirmation
        persistence_check = 1 if (current_data['close'].iloc[-1] / current_data['close'].iloc[-3] - 1) > 0 else -1
        factor_3 = base_signal_3 * persistence_check
        
        # 4. Price Range Compression with Breakout Potential
        # Range Compression Analysis
        range_5d = [(current_data['high'].iloc[j] - current_data['low'].iloc[j]) for j in range(i-4, i+1)]
        range_20d = [(current_data['high'].iloc[j] - current_data['low'].iloc[j]) for j in range(i-19, i+1)]
        avg_range_5d = np.mean(range_5d)
        avg_range_20d = np.mean(range_20d)
        compression_ratio = avg_range_5d / avg_range_20d if avg_range_20d != 0 else 1
        
        # Breakout Probability Indicators
        volume_compression = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-5:].mean()
        price_position = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / high_low_range if high_low_range != 0 else 0.5
        breakout_score = volume_compression * price_position
        
        # Compression-Breakout Signal
        base_factor_4 = compression_ratio * breakout_score
        momentum_direction = current_data['close'].iloc[-1] / current_data['close'].iloc[-2] - 1 if i >= 1 else 0
        factor_4 = base_factor_4 * momentum_direction
        
        # 5. Turnover Velocity with Price Impact
        # Turnover Velocity Calculation
        daily_turnover = current_data['amount'].iloc[-1]
        turnover_5d_avg = current_data['amount'].iloc[-5:].mean()
        turnover_velocity = daily_turnover / turnover_5d_avg if turnover_5d_avg != 0 else 1
        
        # Price Impact Measurement
        absolute_return = abs(current_data['close'].iloc[-1] / current_data['close'].iloc[-2] - 1) if i >= 1 else 0
        volume_weighted_impact = absolute_return * current_data['volume'].iloc[-1]
        impact_efficiency = absolute_return / turnover_velocity if turnover_velocity != 0 else 0
        
        # Velocity-Impact Signal
        base_signal_5 = turnover_velocity * impact_efficiency
        trend_alignment = 1 if (current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1) > 0 else -1
        factor_5 = base_signal_5 * trend_alignment
        
        # Combine all factors with equal weighting
        combined_factor = (factor_1 + factor_2 + factor_3 + factor_4 + factor_5) / 5
        factor.iloc[i] = combined_factor
    
    return factor
