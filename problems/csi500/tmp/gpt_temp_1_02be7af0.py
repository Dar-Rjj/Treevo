import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple market microstructure signals
    with regime-dependent weighting and cross-factor integration.
    """
    data = df.copy()
    
    # Initialize factor components
    momentum_decay = pd.Series(index=data.index, dtype=float)
    regime_signal = pd.Series(index=data.index, dtype=float)
    reversal_prob = pd.Series(index=data.index, dtype=float)
    breakout_conf = pd.Series(index=data.index, dtype=float)
    amount_momentum = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling windows for various components
    for i in range(10, len(data)):
        if i < 2:
            continue
            
        # 1. Intraday Momentum Decay Factor
        current_range = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i-1]
        current_momentum = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        intraday_efficiency = abs(data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i])
        
        if i >= 3:
            prev_momentum = (data['close'].iloc[i-1] - data['close'].iloc[i-2]) / data['close'].iloc[i-2]
            momentum_decay_ratio = current_momentum / prev_momentum if abs(prev_momentum) > 1e-8 else 0
            intraday_persistence = current_momentum / current_range if abs(current_range) > 1e-8 else 0
            
            volume_trend = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
            volume_alignment = np.sign(current_momentum) * np.sign(volume_trend - 1)
            
            momentum_decay.iloc[i] = momentum_decay_ratio * volume_trend * volume_alignment * intraday_persistence
        
        # 2. Volatility Regime Switch Detector
        short_term_vol = (data['high'].iloc[i] - data['low'].iloc[i]) / data['close'].iloc[i-1]
        medium_term_vol = np.mean([(data['high'].iloc[i-j] - data['low'].iloc[i-j]) / data['close'].iloc[i-j-1] 
                                  for j in range(5) if i-j-1 >= 0])
        volatility_ratio = short_term_vol / medium_term_vol if medium_term_vol > 1e-8 else 1
        
        volume_ratio = data['volume'].iloc[i] / np.mean([data['volume'].iloc[i-j] for j in range(1, 5) if i-j >= 0])
        volume_volatility = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
        vol_vol_correlation = np.sign(volatility_ratio - 1) * np.sign(volume_ratio - 1)
        
        transition_strength = volatility_ratio * volume_ratio
        direction_confirmation = vol_vol_correlation * transition_strength
        regime_signal.iloc[i] = direction_confirmation * current_momentum
        
        # 3. Liquidity-Adjusted Price Reversal
        price_position = (data['close'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])
        typical_price = (data['high'].iloc[i] + data['low'].iloc[i] + data['close'].iloc[i]) / 3
        deviation = abs(data['close'].iloc[i] - typical_price) / (data['high'].iloc[i] - data['low'].iloc[i])
        
        effective_spread = (data['high'].iloc[i] - data['low'].iloc[i]) / typical_price
        volume_concentration = data['volume'].iloc[i] / np.mean([data['volume'].iloc[i-j] for j in range(1, 5) if i-j >= 0])
        amount_efficiency = data['amount'].iloc[i] / (data['volume'].iloc[i] * data['close'].iloc[i]) if data['volume'].iloc[i] * data['close'].iloc[i] > 0 else 0
        
        overbought = (price_position > 0.7) * effective_spread
        oversold = (price_position < 0.3) * effective_spread
        reversal_prob.iloc[i] = (overbought - oversold) * volume_concentration * amount_efficiency
        
        # 4. Intraday Range Breakout Confidence
        true_range = max(data['high'].iloc[i] - data['low'].iloc[i],
                        data['high'].iloc[i] - data['close'].iloc[i-1],
                        data['close'].iloc[i-1] - data['low'].iloc[i])
        range_expansion = true_range / np.mean([max(data['high'].iloc[i-j] - data['low'].iloc[i-j],
                                                  data['high'].iloc[i-j] - data['close'].iloc[i-j-1],
                                                  data['close'].iloc[i-j-1] - data['low'].iloc[i-j])
                                              for j in range(1, 10) if i-j-1 >= 0])
        breakout_direction = np.sign(data['close'].iloc[i] - ((data['high'].iloc[i] + data['low'].iloc[i]) / 2))
        
        breakout_volume = data['volume'].iloc[i] / np.mean([data['volume'].iloc[i-j] for j in range(1, 10) if i-j >= 0])
        volume_alignment_breakout = np.sign(range_expansion - 1) * np.sign(breakout_volume - 1)
        volume_persistence = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
        
        breakout_strength = range_expansion * breakout_volume
        validation_score = volume_alignment_breakout * volume_persistence
        breakout_conf.iloc[i] = breakout_strength * validation_score * breakout_direction
        
        # 5. Amount-Based Order Flow Momentum
        up_day_amount = data['amount'].iloc[i] if data['close'].iloc[i] > data['open'].iloc[i] else 0
        down_day_amount = data['amount'].iloc[i] if data['close'].iloc[i] < data['open'].iloc[i] else 0
        net_directional_flow = (up_day_amount - down_day_amount) / (up_day_amount + down_day_amount + 1e-8)
        
        # Consecutive direction count
        consecutive_count = 1
        for j in range(1, min(4, i+1)):
            if (data['close'].iloc[i-j] > data['open'].iloc[i-j]) == (data['close'].iloc[i] > data['open'].iloc[i]):
                consecutive_count += 1
            else:
                break
        
        flow_intensity = abs(net_directional_flow) * consecutive_count
        
        if i >= 3:
            prev_net_flow = (data['amount'].iloc[i-1] if data['close'].iloc[i-1] > data['open'].iloc[i-1] else -data['amount'].iloc[i-1]) / (data['amount'].iloc[i-1] + 1e-8)
            flow_acceleration = net_directional_flow - prev_net_flow
        else:
            flow_acceleration = 0
        
        persistent_flow = flow_intensity * np.sign(net_directional_flow)
        flow_reversal = -flow_acceleration * consecutive_count
        amount_momentum.iloc[i] = persistent_flow + flow_reversal
    
    # Cross-Factor Integration with Regime-Dependent Weighting
    final_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 10:
            continue
            
        # Determine volatility regime
        vol_signal = regime_signal.iloc[i]
        
        if abs(vol_signal) > 1.3:
            # High volatility regime - emphasize reversal and breakout factors
            weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # momentum, regime, reversal, breakout, amount
        elif abs(vol_signal) < 0.7:
            # Low volatility regime - emphasize momentum and amount flow
            weights = [0.3, 0.1, 0.2, 0.2, 0.2]
        else:
            # Normal volatility regime - balanced approach
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Combine factors with regime-dependent weights
        factors = [
            momentum_decay.iloc[i] if not pd.isna(momentum_decay.iloc[i]) else 0,
            regime_signal.iloc[i] if not pd.isna(regime_signal.iloc[i]) else 0,
            reversal_prob.iloc[i] if not pd.isna(reversal_prob.iloc[i]) else 0,
            breakout_conf.iloc[i] if not pd.isna(breakout_conf.iloc[i]) else 0,
            amount_momentum.iloc[i] if not pd.isna(amount_momentum.iloc[i]) else 0
        ]
        
        final_factor.iloc[i] = sum(w * f for w, f in zip(weights, factors))
    
    # Normalize the final factor
    if len(final_factor.dropna()) > 0:
        final_factor = (final_factor - final_factor.mean()) / (final_factor.std() + 1e-8)
    
    return final_factor
