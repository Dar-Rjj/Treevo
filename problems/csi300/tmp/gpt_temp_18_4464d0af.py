import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    data['daily_range'] = data['high'] - data['low']
    data['range_vol_20d'] = data['daily_range'].rolling(window=20).std()
    data['avg_range_vol_20d'] = data['daily_range'].rolling(window=20).mean()
    data['vol_regime'] = np.where(data['daily_range'] > data['avg_range_vol_20d'], 'high', 'low')
    
    # Multi-Timeframe Efficiency Momentum
    # Short-term acceleration (3-5 days)
    data['price_accel'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3) - 
                          (data['close'].shift(3) - data['close'].shift(6)) / data['close'].shift(6))
    data['range_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['vol_adj_momentum'] = data['price_accel'] / (data['high'] - data['low'])
    
    # Medium-term consistency (5-10 days)
    data['dir_persistence'] = (np.sign(data['close'] - data['close'].shift(5)) * 
                              np.sign(data['close'].shift(5) - data['close'].shift(10)))
    
    # Calculate trend efficiency
    price_changes_5d = abs(data['close'] - data['close'].shift(1)).rolling(window=5).sum()
    data['trend_efficiency'] = abs(data['close'] - data['close'].shift(5)) / price_changes_5d
    
    # Pattern stability
    current_sign = np.sign(data['close'] - data['close'].shift(1))
    pattern_stability = []
    for i in range(len(data)):
        if i >= 5:
            window_signs = [np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) 
                          for j in range(i-4, i)]
            stability = sum([1 for sign in window_signs if sign == current_sign.iloc[i]])
            pattern_stability.append(stability)
        else:
            pattern_stability.append(np.nan)
    data['pattern_stability'] = pattern_stability
    
    # Long-term structural momentum (10-20 days)
    data['price_memory'] = data['close'] / data['close'].rolling(window=11).max()
    
    # Support/resistance efficiency
    min_low_10_20 = data['low'].rolling(window=11).min()
    max_high_10_20 = data['high'].rolling(window=11).max()
    data['support_resistance_eff'] = (data['close'] - min_low_10_20) / (max_high_10_20 - min_low_10_20)
    
    # Volume Microstructure Dynamics
    # Multi-scale volume patterns
    data['volume_bursts'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Medium-term accumulation
    volume_accum = []
    for i in range(len(data)):
        if i >= 10:
            window = slice(i-9, i+1)
            signed_volume_sum = sum(data['volume'].iloc[j] * np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) 
                                  for j in range(i-9, i+1))
            total_volume = data['volume'].iloc[i-9:i+1].sum()
            volume_accum.append(signed_volume_sum / total_volume if total_volume != 0 else 0)
        else:
            volume_accum.append(np.nan)
    data['volume_accumulation'] = volume_accum
    
    # Long-term distribution
    data['volume_distribution'] = (data['volume'] / data['volume'].rolling(window=20).mean() - 
                                 data['volume'].shift(10) / data['volume'].shift(10).rolling(window=20).mean())
    
    # Trade size microstructure
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['avg_trade_size'] = data['avg_trade_size'].replace([np.inf, -np.inf], np.nan)
    
    # Retail sentiment
    retail_sentiment = []
    for i in range(len(data)):
        if i >= 5:
            window_volumes = data['volume'].iloc[i-4:i+1]
            threshold = window_volumes.mean() * 0.5
            retail_count = sum(1 for vol in window_volumes if vol < threshold)
            retail_sentiment.append(retail_count)
        else:
            retail_sentiment.append(np.nan)
    data['retail_sentiment'] = retail_sentiment
    
    # Institutional activity
    institutional_activity = []
    for i in range(len(data)):
        if i >= 5:
            window_amounts = data['amount'].iloc[i-4:i+1]
            window_volumes = data['volume'].iloc[i-4:i+1]
            avg_trade_sizes = window_amounts.values / window_volumes.values
            avg_trade_sizes = np.where(np.isinf(avg_trade_sizes) | np.isnan(avg_trade_sizes), 0, avg_trade_sizes)
            threshold = np.mean(avg_trade_sizes)
            
            institutional_amount = sum(amt for j, amt in enumerate(window_amounts) 
                                     if avg_trade_sizes[j] > threshold)
            total_amount = window_amounts.sum()
            institutional_activity.append(institutional_amount / total_amount if total_amount != 0 else 0)
        else:
            institutional_activity.append(np.nan)
    data['institutional_activity'] = institutional_activity
    
    # Range and Liquidity Microstructure
    # Intraday reversal frequency
    reversal_freq = []
    for i in range(len(data)):
        if i >= 5:
            reversals = 0
            for j in range(i-4, i+1):
                if (data['high'].iloc[j] > data['close'].iloc[j-1] and 
                    data['close'].iloc[j] < data['close'].iloc[j-1]):
                    reversals += 1
            reversal_freq.append(reversals)
        else:
            reversal_freq.append(np.nan)
    data['reversal_frequency'] = reversal_freq
    
    # Buying pressure asymmetry
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['buying_pressure_asymmetry'] = data['buying_pressure'] - data['buying_pressure'].shift(1)
    
    # Mean reversion strength
    data['mean_reversion_strength'] = 1 - abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Liquidity cycles
    data['liquidity_ratio'] = data['volume'] / (data['high'] - data['low'])
    data['liquidity_cycles'] = data['liquidity_ratio'] - data['liquidity_ratio'].rolling(window=5).mean()
    
    # Regime change detection
    data['regime_change'] = (data['close'].rolling(window=5).std() / 
                           data['close'].shift(5).rolling(window=5).std())
    
    # Regime-Adaptive Alpha Synthesis
    # Efficiency-weighted momentum combination
    short_term_momentum = (data['vol_adj_momentum'] * data['range_efficiency'] * data['volume_bursts'])
    medium_term_momentum = (data['trend_efficiency'] * data['dir_persistence'] * data['volume_accumulation'])
    long_term_momentum = (data['support_resistance_eff'] * data['price_memory'] * data['volume_distribution'])
    
    # Base signal combination
    base_signal = (0.4 * short_term_momentum + 
                  0.35 * medium_term_momentum + 
                  0.25 * long_term_momentum)
    
    # Microstructure confirmation
    microstructure_adj = (data['mean_reversion_strength'] * 
                         data['buying_pressure_asymmetry'] * 
                         data['liquidity_cycles'])
    
    # Regime-adaptive signal weighting
    final_alpha = []
    for i in range(len(data)):
        if pd.isna(base_signal.iloc[i]) or pd.isna(microstructure_adj.iloc[i]):
            final_alpha.append(np.nan)
            continue
            
        base_val = base_signal.iloc[i]
        micro_adj = microstructure_adj.iloc[i]
        regime = data['vol_regime'].iloc[i]
        
        if regime == 'high':
            # High volatility regime adjustments
            mom_divergence = (short_term_momentum.iloc[i] - medium_term_momentum.iloc[i])
            regime_signal = base_val * mom_divergence * 1.2
            regime_signal *= data['institutional_activity'].iloc[i] * 1.1
        else:
            # Low volatility regime adjustments
            regime_signal = base_val * data['range_efficiency'].iloc[i] * 0.9
            regime_signal *= data['pattern_stability'].iloc[i] * 1.1
        
        # Apply microstructure adjustments and regime change awareness
        final_val = regime_signal * micro_adj * data['regime_change'].iloc[i]
        final_alpha.append(final_val)
    
    # Create output series
    alpha_series = pd.Series(final_alpha, index=data.index, name='alpha_factor')
    
    return alpha_series
