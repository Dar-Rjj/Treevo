import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 15:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use data up to current day
        
        # Amplitude Compression with Liquidity Shock
        # Price Amplitude Compression
        if i >= 10:
            recent_high_low = current_data.iloc[i-1]['high'] - current_data.iloc[i-1]['low']
            window_10 = current_data.iloc[i-10:i]
            max_high = window_10['high'].max()
            min_low = window_10['low'].min()
            if (max_high - min_low) > 0:
                amplitude_compression = recent_high_low / (max_high - min_low)
            else:
                amplitude_compression = 1.0
        else:
            amplitude_compression = 1.0
        
        # Liquidity Shock
        if i >= 5:
            current_v_to_a = current_data.iloc[i-1]['volume'] / current_data.iloc[i-1]['amount'] if current_data.iloc[i-1]['amount'] > 0 else 0
            window_5 = current_data.iloc[i-5:i]
            v_to_a_ratios = window_5['volume'] / window_5['amount'].replace(0, np.nan)
            median_v_to_a = v_to_a_ratios.median()
            if median_v_to_a > 0:
                liquidity_shock = current_v_to_a / median_v_to_a
            else:
                liquidity_shock = 1.0
        else:
            liquidity_shock = 1.0
        
        # Directional bias
        if i >= 3:
            price_bias = np.sign(current_data.iloc[i-1]['close'] - current_data.iloc[i-3]['close'])
        else:
            price_bias = 1.0
        
        breakout_signal = amplitude_compression * liquidity_shock * price_bias
        
        # Volatility Regime Transition with Volume Profile
        # Volatility Regime Shift
        if i >= 15:
            recent_vol = (current_data.iloc[i-8:i]['high'] - current_data.iloc[i-8:i]['low']).std()
            historical_vol = (current_data.iloc[i-15:i-8]['high'] - current_data.iloc[i-15:i-8]['low']).std()
            if historical_vol > 0:
                volatility_shift = recent_vol / historical_vol
            else:
                volatility_shift = 1.0
        else:
            volatility_shift = 1.0
        
        # Volume Distribution Asymmetry
        if i >= 10:
            window_10_vol = current_data.iloc[i-10:i]
            total_volume = window_10_vol['volume'].sum()
            if total_volume > 0:
                upper_volume = 0
                for j in range(i-10, i):
                    high_low_range = current_data.iloc[j]['high'] - current_data.iloc[j]['low']
                    if high_low_range > 0:
                        price_position = (current_data.iloc[j]['close'] - current_data.iloc[j]['low']) / high_low_range
                        if price_position > 0.7:
                            upper_volume += current_data.iloc[j]['volume']
                volume_asymmetry = upper_volume / total_volume
            else:
                volume_asymmetry = 0.5
        else:
            volume_asymmetry = 0.5
        
        # Price range expansion
        if i >= 5:
            recent_range = current_data.iloc[i-1]['high'] - current_data.iloc[i-1]['low']
            historical_range = current_data.iloc[i-5]['high'] - current_data.iloc[i-5]['low']
            if historical_range > 0:
                range_expansion = recent_range / historical_range
            else:
                range_expansion = 1.0
        else:
            range_expansion = 1.0
        
        regime_transition = volatility_shift * volume_asymmetry * range_expansion
        
        # Price Elasticity with Order Flow Imbalance
        # Price Response Elasticity
        if i >= 10:
            price_elasticity = 0
            count = 0
            for j in range(i-10, i):
                if j > 0 and current_data.iloc[j]['volume'] > 0:
                    daily_return = abs(current_data.iloc[j]['close'] / current_data.iloc[j-1]['close'] - 1)
                    price_elasticity += daily_return / current_data.iloc[j]['volume']
                    count += 1
            if count > 0:
                price_elasticity /= count
            else:
                price_elasticity = 0
        else:
            price_elasticity = 0
        
        # Order Flow Imbalance
        if i >= 5:
            window_5_of = current_data.iloc[i-5:i]
            amount_volume_ratios = window_5_of['amount'] / window_5_of['volume'].replace(0, np.nan)
            max_av_ratio = amount_volume_ratios.max()
            mean_av_ratio = amount_volume_ratios.mean()
            if mean_av_ratio > 0:
                order_flow_imbalance = max_av_ratio / mean_av_ratio
            else:
                order_flow_imbalance = 1.0
        else:
            order_flow_imbalance = 1.0
        
        # Recent volatility for adjustment
        if i >= 5:
            returns = []
            for j in range(i-4, i+1):
                if j > 0:
                    returns.append(current_data.iloc[j]['close'] / current_data.iloc[j-1]['close'] - 1)
            recent_volatility = pd.Series(returns).std() if len(returns) > 1 else 0.01
        else:
            recent_volatility = 0.01
        
        microstructure_signal = price_elasticity * order_flow_imbalance / max(recent_volatility, 0.01)
        
        # Combine all factors with equal weighting
        final_factor = (breakout_signal + regime_transition + microstructure_signal) / 3
        result.iloc[i] = final_factor
    
    return result
