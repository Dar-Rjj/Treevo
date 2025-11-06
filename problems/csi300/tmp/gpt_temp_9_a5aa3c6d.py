import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Price-Momentum with Microstructure Divergence factor
    """
    data = df.copy()
    
    # Historical Volatility Estimation (20-day ATR)
    high_20 = data['high'].rolling(window=20, min_periods=10).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan, raw=True)
    low_20 = data['low'].rolling(window=20, min_periods=10).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan, raw=True)
    close_prev = data['close'].shift(1)
    tr = np.maximum(data['high'] - data['low'], 
                   np.maximum(abs(data['high'] - close_prev), 
                             abs(data['low'] - close_prev)))
    atr_20 = tr.rolling(window=20, min_periods=10).mean()
    
    # Volatility Regime Classification
    vol_percentile = atr_20.rolling(window=5, min_periods=3).apply(
        lambda x: np.mean(x < x[-1]) if len(x) > 0 else np.nan, raw=True
    )
    
    # Regime-specific lookback periods
    lookback = np.where(vol_percentile > 0.7, 3, 
                       np.where(vol_percentile < 0.3, 10, 5))
    
    # Regime-Adaptive Momentum Calculation
    momentum_components = []
    for i in range(len(data)):
        if i >= max(lookback):
            lb = lookback[i]
            if i >= lb:
                # Price Momentum Component
                price_momentum = (data['close'].iloc[i] - data['close'].iloc[i-lb]) / data['close'].iloc[i-lb]
                
                # Volume-Weighted Momentum
                vol_window = data['volume'].iloc[i-lb:i+1]
                price_window = data['close'].iloc[i-lb:i+1]
                if len(vol_window) > 0 and vol_window.sum() > 0:
                    vol_weighted_return = np.sum((price_window.diff().fillna(0) * vol_window) / vol_window.sum())
                else:
                    vol_weighted_return = 0
                
                # Acceleration Detection
                if i >= lb * 2:
                    prev_momentum = (data['close'].iloc[i-lb] - data['close'].iloc[i-2*lb]) / data['close'].iloc[i-2*lb]
                    acceleration = price_momentum - prev_momentum
                else:
                    acceleration = 0
                
                momentum_score = price_momentum * 0.6 + vol_weighted_return * 0.3 + acceleration * 0.1
                momentum_components.append(momentum_score)
            else:
                momentum_components.append(0)
        else:
            momentum_components.append(0)
    
    momentum_series = pd.Series(momentum_components, index=data.index)
    
    # Microstructure Divergence Analysis
    divergence_components = []
    for i in range(len(data)):
        if i >= 5:
            # Price-Volume Divergence
            price_trend = (data['close'].iloc[i] - data['close'].iloc[i-5]) / data['close'].iloc[i-5]
            volume_trend = (data['volume'].iloc[i] - data['volume'].iloc[i-5]) / (data['volume'].iloc[i-5] + 1e-8)
            
            if (price_trend > 0 and volume_trend < -0.1) or (price_trend < 0 and volume_trend > 0.1):
                price_volume_div = abs(price_trend) * abs(volume_trend) * -1
            else:
                price_volume_div = abs(price_trend) * abs(volume_trend)
            
            # Bid-Ask Flow Imbalance
            if data['high'].iloc[i] != data['low'].iloc[i]:
                flow_direction = ((data['close'].iloc[i] - data['low'].iloc[i]) / 
                                (data['high'].iloc[i] - data['low'].iloc[i] + 1e-8)) * data['volume'].iloc[i]
                avg_flow = flow_direction / (data['volume'].iloc[i] + 1e-8)
            else:
                avg_flow = 0
            
            # Divergence Strength
            recent_vol = atr_20.iloc[i] if not pd.isna(atr_20.iloc[i]) else 0.01
            divergence_strength = (price_volume_div * 0.7 + avg_flow * 0.3) * recent_vol
            
            # 3-day persistence
            if i >= 7:
                prev_div = np.mean([divergence_components[j] for j in range(i-3, i)])
                divergence_strength = divergence_strength * 0.7 + prev_div * 0.3
            
            divergence_components.append(divergence_strength)
        else:
            divergence_components.append(0)
    
    divergence_series = pd.Series(divergence_components, index=data.index)
    
    # Cross-Timeframe Confirmation
    timeframe_scores = []
    for i in range(len(data)):
        if i >= 8:
            # Intraday momentum
            intraday_momentum = (data['close'].iloc[i] - data['open'].iloc[i]) / (data['open'].iloc[i] + 1e-8)
            
            # Short-term momentum (3-day)
            if i >= 3:
                short_momentum = (data['close'].iloc[i] - data['close'].iloc[i-3]) / data['close'].iloc[i-3]
            else:
                short_momentum = 0
            
            # Medium-term momentum (8-day)
            medium_momentum = (data['close'].iloc[i] - data['close'].iloc[i-8]) / data['close'].iloc[i-8]
            
            # Convergence score
            signals = [intraday_momentum, short_momentum, medium_momentum]
            positive_signals = sum(1 for s in signals if s > 0)
            negative_signals = sum(1 for s in signals if s < 0)
            
            if positive_signals >= 2:
                convergence_score = np.mean([s for s in signals if s > 0])
            elif negative_signals >= 2:
                convergence_score = np.mean([s for s in signals if s < 0])
            else:
                convergence_score = 0
            
            # Overnight gap adjustment
            if i >= 1:
                overnight_gap = (data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                gap_probability = 1 - min(abs(overnight_gap) * 10, 0.8)
            else:
                gap_probability = 1
            
            timeframe_score = convergence_score * gap_probability
            timeframe_scores.append(timeframe_score)
        else:
            timeframe_scores.append(0)
    
    timeframe_series = pd.Series(timeframe_scores, index=data.index)
    
    # Volume Validation
    volume_confidence = []
    for i in range(len(data)):
        if i >= 10:
            current_vol = data['volume'].iloc[i]
            avg_vol_10 = data['volume'].iloc[i-10:i].mean()
            if avg_vol_10 > 0:
                vol_ratio = min(current_vol / avg_vol_10, 3)
                vol_confidence = 1 - abs(1 - vol_ratio) * 0.5
            else:
                vol_confidence = 0.5
        else:
            vol_confidence = 0.5
        volume_confidence.append(max(vol_confidence, 0.1))
    
    volume_series = pd.Series(volume_confidence, index=data.index)
    
    # Final Factor Construction
    final_factor = (momentum_series * divergence_series * timeframe_series * volume_series).fillna(0)
    
    # Apply regime-specific weighting
    regime_weighted_factor = []
    for i in range(len(final_factor)):
        if vol_percentile.iloc[i] > 0.7:  # High volatility
            weight = 1.2
        elif vol_percentile.iloc[i] < 0.3:  # Low volatility
            weight = 0.8
        else:  # Medium volatility
            weight = 1.0
        regime_weighted_factor.append(final_factor.iloc[i] * weight)
    
    return pd.Series(regime_weighted_factor, index=data.index)
