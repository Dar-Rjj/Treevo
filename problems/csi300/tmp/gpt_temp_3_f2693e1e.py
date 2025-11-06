import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum with Microstructure Noise Filtering
    """
    data = df.copy()
    
    # Volatility Regime Identification
    # Daily range volatility
    daily_range = data['high'] - data['low']
    vol_5day_avg = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Volatility regime classification
    high_vol_threshold = 1.5 * vol_5day_avg
    low_vol_threshold = 0.7 * vol_5day_avg
    
    volatility_regime = pd.Series('normal', index=data.index)
    volatility_regime[daily_range > high_vol_threshold] = 'high'
    volatility_regime[daily_range < low_vol_threshold] = 'low'
    
    # Regime-specific lookback periods
    def get_regime_lookback(regime):
        if regime == 'high':
            return {'ultra_short': 1, 'short': 2, 'medium': 3}
        elif regime == 'low':
            return {'ultra_short': 2, 'short': 5, 'medium': 8}
        else:
            return {'ultra_short': 1, 'short': 3, 'medium': 5}
    
    # Microstructure Noise Filtering
    # Bid-ask spread proxy
    mid_price = (data['high'] + data['low']) / 2
    effective_spread = 2 * abs(data['close'] - mid_price) / mid_price
    relative_spread = effective_spread / daily_range
    
    # Price impact measurement (simplified VWAP proxy)
    # Using daily OHLCV to estimate VWAP proxy
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap_proxy = (typical_price * data['volume']).rolling(window=3, min_periods=1).sum() / \
                 data['volume'].rolling(window=3, min_periods=1).sum()
    price_vwap_divergence = abs(data['close'] - vwap_proxy) / vwap_proxy
    
    # Noise threshold identification
    noise_threshold = relative_spread.rolling(window=10, min_periods=5).quantile(0.7)
    high_noise_period = (relative_spread > noise_threshold) | (price_vwap_divergence > price_vwap_divergence.rolling(window=10).quantile(0.7))
    
    # Multi-scale Momentum Convergence
    momentum_signals = pd.DataFrame(index=data.index)
    
    # Ultra-short term momentum (intraday)
    morning_momentum = (data['high'] - data['open']) / data['open']
    afternoon_momentum = (data['close'] - data['low']) / data['low']
    session_divergence = abs(morning_momentum - afternoon_momentum)
    
    # Short-term momentum (daily)
    overnight_momentum = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    intraday_momentum = (data['close'] - data['open']) / data['open']
    overnight_intraday_alignment = np.sign(overnight_momentum) == np.sign(intraday_momentum)
    
    # Medium-term momentum
    momentum_3day = data['close'].pct_change(periods=3)
    momentum_5day = data['close'].pct_change(periods=5)
    price_acceleration = momentum_5day - momentum_3day.shift(2)
    
    # Momentum convergence scoring
    momentum_alignment = pd.DataFrame(index=data.index)
    for i, (idx, row) in enumerate(data.iterrows()):
        regime = volatility_regime.loc[idx]
        lookback = get_regime_lookback(regime)
        
        # Ultra-short term alignment
        ultra_short_pos = morning_momentum.loc[idx] > 0 and afternoon_momentum.loc[idx] > 0
        ultra_short_neg = morning_momentum.loc[idx] < 0 and afternoon_momentum.loc[idx] < 0
        ultra_short_aligned = 1 if ultra_short_pos else (-1 if ultra_short_neg else 0)
        
        # Short-term alignment
        short_term_aligned = 1 if overnight_intraday_alignment.loc[idx] and intraday_momentum.loc[idx] > 0 else \
                            (-1 if overnight_intraday_alignment.loc[idx] and intraday_momentum.loc[idx] < 0 else 0)
        
        # Medium-term alignment
        medium_pos = momentum_3day.loc[idx] > 0 and momentum_5day.loc[idx] > 0
        medium_neg = momentum_3day.loc[idx] < 0 and momentum_5day.loc[idx] < 0
        medium_aligned = 1 if medium_pos else (-1 if medium_neg else 0)
        
        # Convergence score
        alignment_scores = [ultra_short_aligned, short_term_aligned, medium_aligned]
        positive_count = sum(1 for score in alignment_scores if score > 0)
        negative_count = sum(1 for score in alignment_scores if score < 0)
        
        if positive_count > negative_count:
            convergence_score = positive_count / 3.0
        elif negative_count > positive_count:
            convergence_score = -negative_count / 3.0
        else:
            convergence_score = 0
            
        momentum_alignment.loc[idx, 'convergence'] = convergence_score
    
    # Volume-Price Efficiency Divergence
    # Smart volume detection
    volume_20day_median = data['volume'].rolling(window=20, min_periods=10).median()
    volume_spike = data['volume'] > (2 * volume_20day_median)
    
    # Volume persistence
    volume_3day_avg = data['volume'].rolling(window=3, min_periods=2).mean()
    volume_acceleration = (data['volume'] / volume_3day_avg) - 1
    
    # Price efficiency under volume pressure
    normalized_price_impact = abs(intraday_momentum) / (data['volume'] / volume_20day_median)
    historical_efficiency = normalized_price_impact.rolling(window=20, min_periods=10).median()
    efficiency_breakdown = normalized_price_impact > (1.5 * historical_efficiency)
    
    # Volume-efficiency signals
    volume_efficiency_score = pd.Series(0.0, index=data.index)
    
    # Efficient moves: high volume with appropriate price impact
    efficient_bullish = volume_spike & (intraday_momentum > 0) & ~efficiency_breakdown
    efficient_bearish = volume_spike & (intraday_momentum < 0) & ~efficiency_breakdown
    
    # Inefficient moves: high volume with small price change (absorption)
    absorption = volume_spike & (abs(intraday_momentum) < 0.005)
    
    # Illiquid moves: low volume with large price change
    low_volume = data['volume'] < (0.7 * volume_20day_median)
    illiquid_move = low_volume & (abs(intraday_momentum) > 0.02)
    
    volume_efficiency_score[efficient_bullish] = 1.0
    volume_efficiency_score[efficient_bearish] = -1.0
    volume_efficiency_score[absorption] = -0.5 * np.sign(intraday_momentum)
    volume_efficiency_score[illiquid_move] = 0.3 * np.sign(intraday_momentum)
    
    # Adaptive Composite Factor
    final_factor = pd.Series(0.0, index=data.index)
    
    for i, (idx, row) in enumerate(data.iterrows()):
        regime = volatility_regime.loc[idx]
        noise_level = high_noise_period.loc[idx]
        momentum_score = momentum_alignment.loc[idx, 'convergence']
        volume_score = volume_efficiency_score.loc[idx]
        
        # Noise filtering
        noise_discount = 0.3 if noise_level else 1.0
        
        # Regime-weighted combination
        if regime == 'high':
            # Emphasize ultra-short term in high volatility
            regime_weight = 0.6
            ultra_short_weight = 0.4
            base_momentum = (morning_momentum.loc[idx] + afternoon_momentum.loc[idx]) / 2
            momentum_component = (regime_weight * momentum_score + ultra_short_weight * base_momentum)
            
        elif regime == 'low':
            # Emphasize medium-term convergence in low volatility
            regime_weight = 0.8
            medium_weight = 0.2
            momentum_component = (regime_weight * momentum_score + 
                                medium_weight * momentum_5day.loc[idx])
            
        else:  # normal volatility
            # Balanced approach
            momentum_component = momentum_score
        
        # Volume confirmation
        if abs(volume_score) > 0.5:
            volume_confirmation = 1.2  # Enhance signals with volume confirmation
        elif abs(volume_score) < 0.1:
            volume_confirmation = 0.7  # Penalize signals without volume
        else:
            volume_confirmation = 1.0
        
        # Final factor calculation
        final_factor.loc[idx] = (momentum_component * noise_discount * volume_confirmation)
    
    # Normalize the final factor
    factor_std = final_factor.rolling(window=20, min_periods=10).std()
    final_factor_normalized = final_factor / factor_std.replace(0, 1)
    
    return final_factor_normalized
