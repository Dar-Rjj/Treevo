import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_amount'] = df['amount'].shift(1)
    df['prev2_close'] = df['close'].shift(2)
    df['prev2_open'] = df['open'].shift(2)
    
    # Calculate rolling windows for historical data only
    for i in range(len(df)):
        if i < 20:  # Need enough data for calculations
            continue
            
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Asymmetric Return Components
        # Intraday vs Overnight Asymmetry
        intraday_3d = 0
        overnight_3d = 0
        intraday_5d = 0
        overnight_5d = 0
        
        for j in range(max(0, i-4), i+1):
            if j >= 0:
                intraday_ret = current_data.iloc[j]['close'] / current_data.iloc[j]['open'] - 1
                if j > 0:
                    overnight_ret = current_data.iloc[j]['open'] / current_data.iloc[j-1]['close'] - 1
                else:
                    overnight_ret = 0
                
                if j >= i-2:  # Last 3 days
                    intraday_3d += intraday_ret
                    overnight_3d += overnight_ret
                
                if j >= i-4:  # Last 5 days
                    intraday_5d += intraday_ret
                    overnight_5d += overnight_ret
        
        intraday_overnight_asymmetry = (intraday_3d - overnight_3d) - (intraday_5d - overnight_5d)
        
        # Price Efficiency Asymmetry
        price_efficiency_3d = 0
        vol_efficiency_3d = 0
        
        for j in range(max(0, i-2), i+1):
            if j >= 0:
                price_move = current_data.iloc[j]['close'] - current_data.iloc[j]['open']
                high_low_range = current_data.iloc[j]['high'] - current_data.iloc[j]['low']
                volume = current_data.iloc[j]['volume']
                amount = current_data.iloc[j]['amount']
                
                if amount > 0:
                    price_efficiency_3d += price_move * volume / amount
                    vol_efficiency_3d += volume * high_low_range / amount
        
        price_efficiency_asymmetry = price_efficiency_3d - vol_efficiency_3d
        
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
        volume_regime = current_data.iloc[i]['volume'] / volume_median if volume_median > 0 else 1
        
        # Efficiency Regime
        if i >= 5:
            daily_efficiency = abs(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close']) / (current_data.iloc[i]['high'] - current_data.iloc[i]['low'])
            
            high_5d = current_data.iloc[max(0, i-4):i+1]['high'].max()
            low_5d = current_data.iloc[max(0, i-4):i+1]['low'].min()
            range_5d = high_5d - low_5d
            weekly_efficiency = abs(current_data.iloc[i]['close'] - current_data.iloc[i-5]['close']) / range_5d if range_5d > 0 else 0
            
            efficiency_regime = daily_efficiency - weekly_efficiency
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
        if i >= 1:
            price_move = current_data.iloc[i]['close'] - current_data.iloc[i]['open']
            overnight_gap = abs(current_data.iloc[i]['open'] - current_data.iloc[i-1]['close'])
            
            current_vwap = current_data.iloc[i]['amount'] / current_data.iloc[i]['volume'] if current_data.iloc[i]['volume'] > 0 else 0
            prev_vwap = current_data.iloc[i-1]['amount'] / current_data.iloc[i-1]['volume'] if current_data.iloc[i-1]['volume'] > 0 else 0
            
            vwap_change_pct = abs(current_vwap - prev_vwap) / prev_vwap if prev_vwap > 0 else 0
            
            trade_quality = (price_move / overnight_gap) * (1 / (1 + vwap_change_pct)) if overnight_gap > 0 else 0
        else:
            trade_quality = 0
        
        # Alignment Strength
        if i >= 1:
            price_change_pct = abs(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close']) / current_data.iloc[i]['close']
            alignment_strength = direction_alignment * trade_quality * price_change_pct
        else:
            alignment_strength = 0
        
        # Multi-Timeframe Confirmation
        # Intraday Momentum
        high_low_range = current_data.iloc[i]['high'] - current_data.iloc[i]['low']
        intraday_momentum = (current_data.iloc[i]['close'] - current_data.iloc[i]['open']) / high_low_range if high_low_range > 0 else 0
        
        # Overnight Momentum
        if i >= 1:
            close_change = abs(current_data.iloc[i]['close'] - current_data.iloc[i-1]['close'])
            overnight_momentum = (current_data.iloc[i]['open'] - current_data.iloc[i-1]['close']) / close_change if close_change > 0 else 0
        else:
            overnight_momentum = 0
        
        # Momentum Consistency
        momentum_consistency = intraday_momentum * overnight_momentum
        
        # Adaptive Alpha Synthesis
        # Core Asymmetry Signal
        core_asymmetry_signal = intraday_overnight_asymmetry * price_efficiency_asymmetry * asymmetric_momentum
        
        # Regime-Adaptive Weight
        regime_adaptive_weight = 1 + market_state
        
        # Quality-Enhanced Signal
        quality_enhanced_signal = core_asymmetry_signal * alignment_strength
        
        # Final Alpha
        final_alpha = quality_enhanced_signal * regime_adaptive_weight * momentum_consistency
        
        result.iloc[i] = final_alpha
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
