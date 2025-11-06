import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required periods for calculations
    min_periods = max(20, 5)  # For medium-term momentum
    
    for i in range(len(df)):
        if i < min_periods:
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Multi-Scale Momentum Efficiency
        # Short-term momentum efficiency (5-day return / amount flow)
        if i >= 5:
            short_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1)
            short_amount_flow = current_data['amount'].iloc[i-4:i+1].sum()
            short_momentum_eff = short_return / short_amount_flow if short_amount_flow != 0 else 0
        else:
            short_momentum_eff = 0
            
        # Medium-term momentum efficiency (20-day return / amount flow)
        if i >= 20:
            medium_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1)
            medium_amount_flow = current_data['amount'].iloc[i-19:i+1].sum()
            medium_momentum_eff = medium_return / medium_amount_flow if medium_amount_flow != 0 else 0
        else:
            medium_momentum_eff = 0
            
        # Fractal momentum divergence
        momentum_divergence = short_momentum_eff - medium_momentum_eff if i >= 20 else 0
        
        # Volume-Price-Flow Coherence
        # Volume clustering at efficiency boundaries
        current_volume = current_data['volume'].iloc[i]
        vol_5day_avg = current_data['volume'].iloc[max(0, i-4):i+1].mean()
        volume_ratio = current_volume / vol_5day_avg if vol_5day_avg != 0 else 1
        
        # Price-amount flow correlation (3-day)
        if i >= 3:
            price_changes = current_data['close'].iloc[i-2:i+1].pct_change().dropna()
            amount_flows = current_data['amount'].iloc[i-2:i+1]
            if len(price_changes) >= 2 and len(amount_flows) >= 2:
                price_amount_corr = np.corrcoef(price_changes, amount_flows[:len(price_changes)])[0,1]
                price_amount_corr = 0 if np.isnan(price_amount_corr) else price_amount_corr
            else:
                price_amount_corr = 0
        else:
            price_amount_corr = 0
            
        # Flow efficiency: |price change| / amount flow
        if i >= 1:
            price_change = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1])
            daily_amount = current_data['amount'].iloc[i]
            flow_efficiency = price_change / daily_amount if daily_amount != 0 else 0
        else:
            flow_efficiency = 0
            
        # Regime-Adaptive Intraday Efficiency
        # Volatility state classification using price ranges
        current_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if i >= 10:
            avg_range_10day = (current_data['high'].iloc[i-9:i+1] - current_data['low'].iloc[i-9:i+1]).mean()
            volatility_state = 1 if current_range > avg_range_10day else -1
        else:
            volatility_state = 0
            
        # Intraday efficiency: (Close - Open)/(High - Low) per regime
        intraday_efficiency = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / current_range if current_range != 0 else 0
        
        # Movement efficiency: |Close_t - Close_{t-1}|/TrueRange per regime
        if i >= 1:
            true_range = max(
                current_data['high'].iloc[i] - current_data['low'].iloc[i],
                abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1]),
                abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1])
            )
            movement_efficiency = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / true_range if true_range != 0 else 0
        else:
            movement_efficiency = 0
            
        # Range-Volume Compression Signals
        # Simultaneous range compression and volume efficiency
        if i >= 5:
            range_5day_avg = (current_data['high'].iloc[i-4:i+1] - current_data['low'].iloc[i-4:i+1]).mean()
            range_compression = current_range / range_5day_avg if range_5day_avg != 0 else 1
            
            volume_efficiency_5day = flow_efficiency / current_data['amount'].iloc[i-4:i+1].mean() if current_data['amount'].iloc[i-4:i+1].mean() != 0 else 0
        else:
            range_compression = 1
            volume_efficiency_5day = 0
            
        # Breakout detection with flow coherence confirmation
        breakout_signal = 0
        if i >= 5 and range_compression < 0.8 and volume_ratio > 1.2:
            # Potential breakout condition
            price_trend = current_data['close'].iloc[i] > current_data['close'].iloc[i-5]
            flow_coherence = price_amount_corr > 0.3
            if price_trend and flow_coherence:
                breakout_signal = 1
            elif not price_trend and flow_coherence:
                breakout_signal = -1
                
        # Cross-Dimensional Coherence Analysis
        # Price-amount-liquidity movement coherence
        coherence_score = 0
        if i >= 3:
            price_trend_3d = np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-3])
            amount_trend_3d = np.sign(current_data['amount'].iloc[i] - current_data['amount'].iloc[i-3].mean())
            volume_trend_3d = np.sign(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-3].mean())
            
            coherence_count = sum([
                1 if price_trend_3d == amount_trend_3d else 0,
                1 if price_trend_3d == volume_trend_3d else 0,
                1 if amount_trend_3d == volume_trend_3d else 0
            ])
            coherence_score = coherence_count / 3.0
            
        # Combine all components with appropriate weights
        factor_value = (
            0.15 * short_momentum_eff +
            0.12 * medium_momentum_eff +
            0.10 * momentum_divergence +
            0.08 * volume_ratio +
            0.10 * price_amount_corr +
            0.08 * flow_efficiency +
            0.07 * volatility_state * intraday_efficiency +
            0.07 * volatility_state * movement_efficiency +
            0.08 * breakout_signal +
            0.15 * coherence_score
        )
        
        result.iloc[i] = factor_value
    
    return result
