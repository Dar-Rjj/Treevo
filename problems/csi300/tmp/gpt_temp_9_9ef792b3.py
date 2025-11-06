import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum-Volume Synchronization with Volatility-Adaptive Persistence
    """
    data = df.copy()
    
    # Calculate mid-price
    data['mid'] = (data['high'] + data['low']) / 2
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    for i in range(max(30, len(data))):
        if i < 30:
            continue
            
        current_data = data.iloc[:i+1]
        current_day = current_data.iloc[-1]
        
        # 1. Intraday Momentum-Volume Synchronization
        # For simplicity, we'll use full day data as proxy for intraday sessions
        # In practice, you would need intraday data for morning/afternoon sessions
        
        # Morning session proxy (first half of trading range)
        high_morning = current_day['high'] * 0.8 + current_day['open'] * 0.2
        low_morning = current_day['low'] * 0.8 + current_day['open'] * 0.2
        close_morning = (current_day['high'] + current_day['low']) / 2
        
        morning_momentum_asymmetry = ((high_morning - current_day['open']) / current_day['open'] - 
                                    (current_day['open'] - low_morning) / current_day['open'])
        
        if (high_morning - low_morning) > 0:
            morning_volume_flow = ((close_morning - current_day['open']) / 
                                 (high_morning - low_morning)) * current_day['volume']
        else:
            morning_volume_flow = 0
            
        morning_sync = np.sign(morning_momentum_asymmetry) * np.sign(morning_volume_flow)
        morning_strength = abs(morning_momentum_asymmetry) * abs(morning_volume_flow)
        
        # Afternoon session proxy (second half of trading range)
        high_afternoon = current_day['high']
        low_afternoon = current_day['low']
        open_afternoon = (current_day['high'] + current_day['low']) / 2
        
        afternoon_momentum_asymmetry = ((current_day['close'] - low_afternoon) / current_day['close'] - 
                                      (high_afternoon - current_day['close']) / current_day['close'])
        
        if (high_afternoon - low_afternoon) > 0:
            afternoon_volume_flow = ((current_day['close'] - open_afternoon) / 
                                   (high_afternoon - low_afternoon)) * current_day['volume']
        else:
            afternoon_volume_flow = 0
            
        afternoon_sync = np.sign(afternoon_momentum_asymmetry) * np.sign(afternoon_volume_flow)
        afternoon_strength = abs(afternoon_momentum_asymmetry) * abs(afternoon_volume_flow)
        
        # Full-day persistence scoring
        session_momentum_corr = np.sign(morning_momentum_asymmetry * afternoon_momentum_asymmetry)
        session_volume_corr = np.sign(morning_volume_flow * afternoon_volume_flow)
        
        # Cross-session synchronization persistence
        persistence_score = 0
        if morning_sync > 0 and afternoon_sync > 0:
            persistence_score = 2
        elif morning_sync > 0 or afternoon_sync > 0:
            persistence_score = 1
        elif morning_sync < 0 and afternoon_sync < 0:
            persistence_score = -2
        elif morning_sync < 0 or afternoon_sync < 0:
            persistence_score = -1
            
        # 2. Volatility-Adaptive Multi-Timeframe Analysis
        # Calculate volatility regime
        recent_returns = current_data['mid'].pct_change().dropna().tail(30)
        upside_vol = recent_returns[recent_returns > 0].std() if len(recent_returns[recent_returns > 0]) > 1 else 0
        downside_vol = abs(recent_returns[recent_returns < 0].std()) if len(recent_returns[recent_returns < 0]) > 1 else 0
        
        if downside_vol > 0:
            volatility_asymmetry = (upside_vol / downside_vol) - 1
        else:
            volatility_asymmetry = 0
            
        # Volatility regime classification
        if volatility_asymmetry > 0.2:
            volatility_regime = 'bull'
        elif volatility_asymmetry < -0.2:
            volatility_regime = 'bear'
        else:
            volatility_regime = 'neutral'
            
        # Multi-timeframe synchronization
        timeframe_scores = []
        timeframe_strengths = []
        
        # Short-term (5-day)
        if i >= 5:
            mid_5d_return = (current_day['mid'] - current_data.iloc[-6]['mid']) / current_data.iloc[-6]['mid']
            vol_5d = current_data['mid'].pct_change().tail(5).std()
            vol_momentum_5d = current_data['volume'].tail(5).pct_change().mean()
            
            if vol_5d > 0:
                adj_return_5d = mid_5d_return / vol_5d
            else:
                adj_return_5d = mid_5d_return
                
            short_term_sync = np.sign(adj_return_5d) * np.sign(vol_momentum_5d)
            short_term_strength = abs(adj_return_5d) * abs(vol_momentum_5d)
            timeframe_scores.append(short_term_sync)
            timeframe_strengths.append(short_term_strength)
        
        # Medium-term (10-day)
        if i >= 10:
            mid_10d_return = (current_day['mid'] - current_data.iloc[-11]['mid']) / current_data.iloc[-11]['mid']
            vol_10d = current_data['mid'].pct_change().tail(10).std()
            vol_momentum_10d = current_data['volume'].tail(10).pct_change().mean()
            
            if vol_10d > 0:
                adj_return_10d = mid_10d_return / vol_10d
            else:
                adj_return_10d = mid_10d_return
                
            medium_term_sync = np.sign(adj_return_10d) * np.sign(vol_momentum_10d)
            medium_term_strength = abs(adj_return_10d) * abs(vol_momentum_10d)
            timeframe_scores.append(medium_term_sync)
            timeframe_strengths.append(medium_term_strength)
        
        # Multi-timeframe persistence scoring
        if timeframe_scores:
            positive_count = sum(1 for score in timeframe_scores if score > 0)
            negative_count = sum(1 for score in timeframe_scores if score < 0)
            net_sync = positive_count - negative_count
            
            avg_strength = np.mean(timeframe_strengths) if timeframe_strengths else 0
        else:
            net_sync = 0
            avg_strength = 0
            
        # 3. Volume-Price Efficiency and Breakout Analysis
        # Breakout efficiency
        prev_high_5d = current_data['high'].tail(6).head(5).max()
        prev_low_5d = current_data['low'].tail(6).head(5).min()
        
        true_range = max(current_day['high'] - current_day['low'],
                        abs(current_day['high'] - current_data.iloc[-2]['close']),
                        abs(current_day['low'] - current_data.iloc[-2]['close']))
        
        breakout_magnitude = (max(0, current_day['high'] - prev_high_5d) + 
                            max(0, prev_low_5d - current_day['low']))
        
        if true_range > 0:
            efficiency_ratio = breakout_magnitude / true_range
        else:
            efficiency_ratio = 0
            
        # Daily price efficiency
        if (current_day['high'] - current_day['low']) > 0:
            price_efficiency = (current_day['close'] - current_day['open']) / (current_day['high'] - current_day['low'])
        else:
            price_efficiency = 0
            
        volume_weighted_efficiency = price_efficiency * current_day['volume']
        
        # 4. Composite Alpha Generation
        # Base synchronization score
        base_score = (persistence_score * 0.4 + 
                     net_sync * 0.3 + 
                     np.sign(price_efficiency) * 0.2 + 
                     np.sign(morning_sync + afternoon_sync) * 0.1)
        
        # Volatility regime adjustment
        if volatility_regime == 'bull':
            base_score *= 1.2
        elif volatility_regime == 'bear':
            base_score *= 0.8
            
        # Strength weighting
        strength_multiplier = 1 + (morning_strength + afternoon_strength + avg_strength) / 3
        base_score *= strength_multiplier
        
        # Efficiency bonus/penalty
        if efficiency_ratio > 0.7 and price_efficiency > 0:
            base_score += 0.5
        elif efficiency_ratio > 0.7 and price_efficiency < 0:
            base_score -= 0.5
            
        # Final alpha value
        alpha.iloc[i] = base_score
        
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
