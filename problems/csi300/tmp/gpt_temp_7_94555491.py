import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all required columns
    data['bullish_reversal'] = 0
    data['bearish_reversal'] = 0
    data['range_reversal_signal'] = 0
    data['short_term_range_momentum'] = 0
    data['medium_term_range_momentum'] = 0
    data['range_momentum_divergence'] = 0
    data['range_based_efficiency'] = 0
    data['multi_day_efficiency'] = 0
    data['efficiency_divergence'] = 0
    data['true_range'] = 0
    data['avg_true_range'] = 0
    data['volatility_persistence'] = 0
    data['volume_spike'] = 0
    data['volume_momentum'] = 0
    data['volume_volatility_ratio'] = 0
    data['range_volatility_concentration'] = 0
    data['opening_gap_volatility'] = 0
    data['session_volatility_bias'] = 0
    data['short_term_price_momentum'] = 0
    data['medium_term_price_momentum'] = 0
    data['intraday_momentum'] = 0
    data['range_price_momentum_alignment'] = 0
    data['efficiency_weighted_momentum'] = 0
    data['divergence_confirmation'] = 0
    data['volume_supported_momentum'] = 0
    data['spike_enhanced_momentum'] = 0
    data['persistence_weighting'] = 0
    
    # Calculate components iteratively to avoid lookahead bias
    for i in range(20, len(data)):
        if i < 20:
            continue
            
        # Asymmetric Range Reversal Detection
        midpoint = (data['high'].iloc[i] + data['low'].iloc[i]) / 2
        bullish_cond = (data['close'].iloc[i] > midpoint and 
                       data['close'].iloc[i] > data['low'].iloc[i-1] and 
                       data['high'].iloc[i] > data['high'].iloc[i-1])
        bearish_cond = (data['close'].iloc[i] < midpoint and 
                       data['close'].iloc[i] < data['high'].iloc[i-1] and 
                       data['low'].iloc[i] < data['low'].iloc[i-1])
        
        data.loc[data.index[i], 'bullish_reversal'] = 1 if bullish_cond else 0
        data.loc[data.index[i], 'bearish_reversal'] = 1 if bearish_cond else 0
        data.loc[data.index[i], 'range_reversal_signal'] = (data['bullish_reversal'].iloc[i] - 
                                                           data['bearish_reversal'].iloc[i])
        
        # Multi-Timeframe Range Momentum
        if i >= 3:
            short_term_range = data['high'].iloc[i] - data['low'].iloc[i]
            short_term_range_3 = data['high'].iloc[i-3] - data['low'].iloc[i-3]
            if short_term_range_3 > 0:
                data.loc[data.index[i], 'short_term_range_momentum'] = (short_term_range / short_term_range_3) - 1
        
        if i >= 10:
            medium_term_range = data['high'].iloc[i] - data['low'].iloc[i]
            medium_term_range_10 = data['high'].iloc[i-10] - data['low'].iloc[i-10]
            if medium_term_range_10 > 0:
                data.loc[data.index[i], 'medium_term_range_momentum'] = (medium_term_range / medium_term_range_10) - 1
        
        data.loc[data.index[i], 'range_momentum_divergence'] = abs(
            data['short_term_range_momentum'].iloc[i] - data['medium_term_range_momentum'].iloc[i]
        )
        
        # Price Movement Efficiency Integration
        if i >= 1:
            daily_range = data['high'].iloc[i] - data['low'].iloc[i]
            if daily_range > 0:
                data.loc[data.index[i], 'range_based_efficiency'] = abs(
                    data['close'].iloc[i] - data['close'].iloc[i-1]
                ) / daily_range
        
        if i >= 5:
            multi_day_range_sum = sum(data['high'].iloc[i-j] - data['low'].iloc[i-j] for j in range(5))
            if multi_day_range_sum > 0:
                data.loc[data.index[i], 'multi_day_efficiency'] = abs(
                    data['close'].iloc[i] - data['close'].iloc[i-5]
                ) / multi_day_range_sum
        
        data.loc[data.index[i], 'efficiency_divergence'] = (
            data['range_based_efficiency'].iloc[i] - data['multi_day_efficiency'].iloc[i]
        )
        
        # True Range Volatility Components
        tr1 = data['high'].iloc[i] - data['low'].iloc[i]
        tr2 = abs(data['high'].iloc[i] - data['close'].iloc[i-1]) if i >= 1 else 0
        tr3 = abs(data['low'].iloc[i] - data['close'].iloc[i-1]) if i >= 1 else 0
        data.loc[data.index[i], 'true_range'] = max(tr1, tr2, tr3)
        
        if i >= 4:
            atr_window = data['true_range'].iloc[i-4:i+1]
            data.loc[data.index[i], 'avg_true_range'] = atr_window.mean()
            if data['avg_true_range'].iloc[i] > 0:
                data.loc[data.index[i], 'volatility_persistence'] = (
                    data['true_range'].iloc[i] / data['avg_true_range'].iloc[i]
                )
        
        # Volume-Volatility Alignment
        if i >= 4:
            volume_window = data['volume'].iloc[i-4:i+1]
            volume_median = volume_window.median()
            data.loc[data.index[i], 'volume_spike'] = 1 if data['volume'].iloc[i] > volume_median else 0
            
            if i >= 3 and data['volume'].iloc[i-3] > 0:
                data.loc[data.index[i], 'volume_momentum'] = (
                    data['volume'].iloc[i] / data['volume'].iloc[i-3] - 1
                )
            
            if data['avg_true_range'].iloc[i] > 0:
                data.loc[data.index[i], 'volume_volatility_ratio'] = (
                    data['volume'].iloc[i] / (data['avg_true_range'].iloc[i] + 1e-8)
                )
        
        # Intraday Volatility Patterns
        if i >= 10:
            range_volatility_values = []
            for j in range(10):
                if i-j >= 0:
                    daily_range_vol = (data['high'].iloc[i-j] - data['low'].iloc[i-j]) / data['close'].iloc[i-j]
                    range_volatility_values.append(daily_range_vol)
            if len(range_volatility_values) >= 2:
                data.loc[data.index[i], 'range_volatility_concentration'] = np.std(range_volatility_values)
        
        daily_range_current = data['high'].iloc[i] - data['low'].iloc[i]
        if daily_range_current > 0 and i >= 1:
            data.loc[data.index[i], 'opening_gap_volatility'] = abs(
                data['open'].iloc[i] - data['close'].iloc[i-1]
            ) / daily_range_current
        
        if daily_range_current > 0:
            data.loc[data.index[i], 'session_volatility_bias'] = (
                (data['high'].iloc[i] - data['open'].iloc[i]) / daily_range_current - 0.5
            )
        
        # Price Momentum Components
        if i >= 5 and data['close'].iloc[i-5] > 0:
            data.loc[data.index[i], 'short_term_price_momentum'] = (
                data['close'].iloc[i] / data['close'].iloc[i-5] - 1
            )
        
        if i >= 20 and data['close'].iloc[i-20] > 0:
            data.loc[data.index[i], 'medium_term_price_momentum'] = (
                data['close'].iloc[i] / data['close'].iloc[i-20] - 1
            )
        
        if i >= 3:
            current_mid = (data['high'].iloc[i] + data['low'].iloc[i]) / 2
            past_mid = (data['high'].iloc[i-3] + data['low'].iloc[i-3]) / 2
            if past_mid > 0:
                data.loc[data.index[i], 'intraday_momentum'] = current_mid / past_mid - 1
        
        # Range-Price Momentum Alignment
        st_range_mom = data['short_term_range_momentum'].iloc[i]
        st_price_mom = data['short_term_price_momentum'].iloc[i]
        data.loc[data.index[i], 'range_price_momentum_alignment'] = np.sign(st_range_mom) * np.sign(st_price_mom)
        
        data.loc[data.index[i], 'efficiency_weighted_momentum'] = (
            st_price_mom * data['range_based_efficiency'].iloc[i]
        )
        
        data.loc[data.index[i], 'divergence_confirmation'] = (
            data['range_momentum_divergence'].iloc[i] * 
            abs(st_price_mom - data['medium_term_price_momentum'].iloc[i])
        )
        
        # Volume-Momentum Confirmation
        data.loc[data.index[i], 'volume_supported_momentum'] = (
            st_price_mom * data['volume_momentum'].iloc[i]
        )
        
        data.loc[data.index[i], 'spike_enhanced_momentum'] = (
            data['volume_spike'].iloc[i] * st_price_mom
        )
        
        # Persistence Weighting (simplified)
        if i >= 4:
            volume_above_median_count = sum(
                1 for j in range(5) if data['volume'].iloc[i-j] > volume_median
            )
            data.loc[data.index[i], 'persistence_weighting'] = (
                volume_above_median_count * data['volume_supported_momentum'].iloc[i]
            )
    
    # Composite Factor Construction
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        # Core Reversal-Range Engine
        range_reversal_base = (
            data['range_reversal_signal'].iloc[i] * data['range_momentum_divergence'].iloc[i]
        )
        efficiency_adjustment = range_reversal_base * data['efficiency_divergence'].iloc[i]
        volume_confirmation = efficiency_adjustment * data['volume_volatility_ratio'].iloc[i]
        
        # Volatility-Weighted Integration
        volatility_normalization = volume_confirmation / (data['avg_true_range'].iloc[i] + 1e-8)
        volatility_persistence_enhancement = (
            volatility_normalization * data['volatility_persistence'].iloc[i]
        )
        concentration_filter = (
            volatility_persistence_enhancement * (1 + data['range_volatility_concentration'].iloc[i])
        )
        
        # Multi-Timeframe Momentum Finalization
        momentum_alignment_layer = (
            concentration_filter * data['range_price_momentum_alignment'].iloc[i]
        )
        volume_momentum_boost = (
            momentum_alignment_layer * data['volume_supported_momentum'].iloc[i]
        )
        final_factor = volume_momentum_boost * data['session_volatility_bias'].iloc[i]
        
        factor_values.iloc[i] = final_factor
    
    # Fill NaN values with 0 for early periods
    factor_values = factor_values.fillna(0)
    
    return factor_values
