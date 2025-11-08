import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required periods
    for i in range(5, len(df)):
        if i < 10:  # Need enough data for calculations
            continue
            
        # Extract current and historical data
        current = df.iloc[i]
        prev1 = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        prev3 = df.iloc[i-3]
        prev4 = df.iloc[i-4]
        prev5 = df.iloc[i-5]
        
        # Multi-Timeframe Momentum Regime
        # Short-term Momentum (1-3 days)
        price_accel = (current['close']/prev1['close'] - 1) - (prev1['close']/prev2['close'] - 1)
        volume_accel = (current['volume']/prev1['volume'] - 1) - (prev1['volume']/prev2['volume'] - 1)
        range_eff = (current['close'] - prev1['close']) / (current['high'] - current['low'])
        short_term_momentum = (price_accel + volume_accel + range_eff) / 3
        
        # Medium-term Momentum (5-10 days)
        price_trend = current['close'] / prev5['close'] - 1
        volume_trend = current['volume'] / prev5['volume'] - 1
        vol_persistence = (current['high'] - current['low']) / (prev5['high'] - prev5['low'])
        medium_term_momentum = (price_trend + volume_trend + vol_persistence) / 3
        
        # Regime Detection
        price_change_5d = abs(current['close']/prev5['close'] - 1)
        daily_range = (current['high'] - current['low'])/current['close']
        
        trending_regime = price_change_5d > daily_range
        mean_reversion_regime = price_change_5d < daily_range
        breakout_regime = (current['close'] > prev1['high']) or (current['close'] < prev1['low'])
        
        # Regime-Adaptive Momentum
        trending_weight = 0.7 if trending_regime else 0.3
        mean_reversion_weight = 0.7 if mean_reversion_regime else 0.3
        regime_adaptive_momentum = (short_term_momentum * trending_weight + 
                                  medium_term_momentum * mean_reversion_weight)
        
        # Volume-Price Efficiency Divergence
        # Price Efficiency Metrics
        intraday_eff = (current['close'] - current['open']) / (current['high'] - current['low'])
        close_to_close_eff = (current['close'] - prev1['close']) / (current['high'] - current['low'])
        gap_eff = (current['open'] - prev1['close']) / (current['high'] - current['low'])
        
        # Volume Anomaly Detection
        short_vol_spike = current['volume'] / ((prev1['volume'] + prev2['volume']) / 2)
        medium_vol_trend = current['volume'] / ((prev3['volume'] + prev4['volume'] + prev5['volume']) / 3)
        vol_volatility = (current['volume']/prev1['volume'] - 1) - (prev1['volume']/prev2['volume'] - 1)
        
        # Efficiency-Volume Divergence
        core_efficiency = (intraday_eff + close_to_close_eff) / 2
        volume_confirmation = np.sqrt(short_vol_spike * medium_vol_trend)
        efficiency_volume_divergence = core_efficiency * volume_confirmation * vol_volatility
        
        # Intraday Session Regime Analysis
        opening_strength = (current['open'] - prev1['close']) / prev1['close']
        morning_trend = (current['high'] - current['open']) / current['open']
        morning_support = (current['open'] - current['low']) / current['open']
        
        afternoon_continuation = (current['close'] - (current['high'] + current['low'])/2) / ((current['high'] + current['low'])/2)
        closing_strength = (current['close'] - current['low']) / (current['high'] - current['low'])
        session_reversal = abs(current['close'] - current['open']) / (current['high'] - current['low'])
        
        # Session Regime Classification
        bullish_session = (morning_trend > 0) and (afternoon_continuation > 0)
        bearish_session = (morning_trend < 0) and (afternoon_continuation < 0)
        mixed_session = (morning_trend * afternoon_continuation) < 0
        
        session_signal = (morning_trend + afternoon_continuation + closing_strength) / 3
        
        # Volatility-Regime Adaptive Signals
        current_vol = (current['high'] - current['low'])/current['close']
        prev5_vol = (prev5['high'] - prev5['low'])/prev5['close']
        vol_change = abs(current_vol - prev5_vol)
        
        low_vol_regime = current_vol < prev5_vol
        high_vol_regime = current_vol > prev5_vol
        transition_regime = vol_change < 0.01
        
        # Regime-Adaptive Blending
        regime_signals = []
        if low_vol_regime:
            regime_signals.append(regime_adaptive_momentum * 0.6)
        if high_vol_regime:
            regime_signals.append(efficiency_volume_divergence * 0.7)
        if transition_regime:
            regime_signals.append(session_signal * 0.5)
        
        volatility_regime_signal = sum(regime_signals) if regime_signals else 0
        
        # Final Alpha Construction
        # Dynamic Weighting
        momentum_weight = 0.4 if trending_regime else 0.2
        
        vol_avg = (df.iloc[i-4:i+1]['volume'].mean())
        volume_weight = 0.3 if current['volume'] > vol_avg else 0.1
        
        if bullish_session:
            intraday_weight = 0.2
        elif bearish_session:
            intraday_weight = 0.1
        else:
            intraday_weight = 0.05
        
        # Regime weight based on volatility
        if low_vol_regime:
            regime_weight = 0.1
        elif high_vol_regime:
            regime_weight = 0.3
        else:
            regime_weight = 0.2
        
        # Weighted combination
        final_signal = (regime_adaptive_momentum * momentum_weight +
                       efficiency_volume_divergence * volume_weight +
                       session_signal * intraday_weight +
                       volatility_regime_signal * regime_weight)
        
        # Volatility scaling and range bounding
        if current_vol > 0:
            final_signal /= current_vol
        
        # Range bounding for outlier control
        final_signal = max(min(final_signal, 2), -2)
        
        result.iloc[i] = final_signal
    
    return result
