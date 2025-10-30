import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_amount'] = df['amount'].shift(1)
    df['prev2_close'] = df['close'].shift(2)
    df['prev2_open'] = df['open'].shift(2)
    
    # Rolling calculations
    for i in range(len(df)):
        if i < 20:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Asymmetric Return Components
        # 3-day components
        intraday_3d = 0
        overnight_3d = 0
        for j in range(max(0, i-2), i+1):
            if j >= 2:  # Need previous close data
                intraday_3d += (current_data.iloc[j]['close'] / current_data.iloc[j]['open'] - 1)
                overnight_3d += (current_data.iloc[j]['open'] / current_data.iloc[j-1]['close'] - 1)
        
        # 5-day components
        intraday_5d = 0
        overnight_5d = 0
        for j in range(max(0, i-4), i+1):
            if j >= 1:  # Need previous close data
                intraday_5d += (current_data.iloc[j]['close'] / current_data.iloc[j]['open'] - 1)
                overnight_5d += (current_data.iloc[j]['open'] / current_data.iloc[j-1]['close'] - 1)
        
        asymmetric_return = (intraday_3d - overnight_3d) - (intraday_5d - overnight_5d)
        
        # Price Efficiency Asymmetry
        price_efficiency_3d = 0
        volatility_efficiency_3d = 0
        for j in range(max(0, i-2), i+1):
            if current_data.iloc[j]['amount'] != 0:
                price_efficiency_3d += (current_data.iloc[j]['close'] - current_data.iloc[j]['open']) * current_data.iloc[j]['volume'] / current_data.iloc[j]['amount']
                volatility_efficiency_3d += current_data.iloc[j]['volume'] * (current_data.iloc[j]['high'] - current_data.iloc[j]['low']) / current_data.iloc[j]['amount']
        
        price_efficiency_asymmetry = price_efficiency_3d - volatility_efficiency_3d
        
        # Asymmetric Momentum
        if i >= 2:
            current_intraday = current_data.iloc[i]['close'] / current_data.iloc[i]['open']
            prev_intraday = current_data.iloc[i-1]['close'] / current_data.iloc[i-1]['open']
            current_overnight = current_data.iloc[i]['open'] / current_data.iloc[i-1]['close']
            prev_overnight = current_data.iloc[i-1]['open'] / current_data.iloc[i-2]['close']
            asymmetric_momentum = (current_intraday - prev_intraday) - (current_overnight - prev_overnight)
        else:
            asymmetric_momentum = 0
        
        # Dynamic Regime Detection
        # Volatility Regime
        volatility_regime = (current_data.iloc[i]['high'] - current_data.iloc[i]['low']) / current_data.iloc[i]['close']
        
        # Volume Regime
        volume_window = current_data.iloc[max(0, i-19):i+1]['volume']
        volume_median = volume_window.median()
        volume_regime = current_data.iloc[i]['volume'] / volume_median if volume_median != 0 else 1
        
        # Efficiency Regime
        if i >= 5:
            recent_eff = abs(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close']) / (current_data.iloc[i]['high'] - current_data.iloc[i]['low'])
            high_5d = current_data.iloc[max(0, i-4):i+1]['high'].max()
            low_5d = current_data.iloc[max(0, i-4):i+1]['low'].min()
            range_5d = high_5d - low_5d
            historical_eff = abs(current_data.iloc[i]['close'] - current_data.iloc[i-5]['close']) / range_5d if range_5d != 0 else 0
            efficiency_regime = recent_eff - historical_eff
        else:
            efficiency_regime = 0
        
        # Market State
        market_state = volatility_regime * volume_regime * efficiency_regime
        
        # Volume-Price Alignment Quality
        # Direction Alignment
        if i >= 1:
            vol_change_sign = np.sign(current_data.iloc[i]['volume'] - current_data.iloc[i-1]['volume'])
            price_change_sign = np.sign(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close'])
            direction_alignment = vol_change_sign * price_change_sign
        else:
            direction_alignment = 0
        
        # Trade Quality
        if i >= 1 and current_data.iloc[i-1]['volume'] != 0:
            price_move_ratio = (current_data.iloc[i]['close'] - current_data.iloc[i]['open']) / abs(current_data.iloc[i]['open'] - current_data.iloc[i-1]['close']) if abs(current_data.iloc[i]['open'] - current_data.iloc[i-1]['close']) != 0 else 0
            
            current_vwap = current_data.iloc[i]['amount'] / current_data.iloc[i]['volume'] if current_data.iloc[i]['volume'] != 0 else 0
            prev_vwap = current_data.iloc[i-1]['amount'] / current_data.iloc[i-1]['volume'] if current_data.iloc[i-1]['volume'] != 0 else 0
            
            vwap_change_ratio = abs(current_vwap - prev_vwap) / prev_vwap if prev_vwap != 0 else 1
            trade_quality = price_move_ratio * (1 / (1 + vwap_change_ratio))
        else:
            trade_quality = 0
        
        # Alignment Strength
        if current_data.iloc[i]['close'] != 0:
            price_change_magnitude = abs(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close']) / current_data.iloc[i]['close'] if i >= 1 else 0
        else:
            price_change_magnitude = 0
        
        alignment_strength = direction_alignment * trade_quality * price_change_magnitude
        
        # Multi-Timeframe Confirmation
        # Intraday Momentum
        intraday_momentum = (current_data.iloc[i]['close'] - current_data.iloc[i]['open']) / (current_data.iloc[i]['high'] - current_data.iloc[i]['low']) if (current_data.iloc[i]['high'] - current_data.iloc[i]['low']) != 0 else 0
        
        # Overnight Momentum
        if i >= 1 and abs(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close']) != 0:
            overnight_momentum = (current_data.iloc[i]['open'] - current_data.iloc[i-1]['close']) / abs(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close'])
        else:
            overnight_momentum = 0
        
        # Momentum Consistency
        momentum_consistency = intraday_momentum * overnight_momentum
        
        # Adaptive Alpha Synthesis
        # Core Asymmetry Signal
        core_asymmetry_signal = asymmetric_return * price_efficiency_asymmetry * asymmetric_momentum
        
        # Regime-Adaptive Weight
        regime_adaptive_weight = 1 + market_state
        
        # Quality-Enhanced Signal
        quality_enhanced_signal = core_asymmetry_signal * alignment_strength
        
        # Final Alpha
        final_alpha = quality_enhanced_signal * regime_adaptive_weight * momentum_consistency
        
        result.iloc[i] = final_alpha
    
    # Clean up temporary columns
    df.drop(['prev_close', 'prev_open', 'prev_volume', 'prev_amount', 'prev2_close', 'prev2_open'], axis=1, inplace=True, errors='ignore')
    
    return result
