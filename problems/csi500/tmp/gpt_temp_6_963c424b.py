import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum Divergence Factor
    Combines multi-timeframe momentum with regime detection and divergence analysis
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    for period in [5, 10, 20]:
        df[f'price_momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
        df[f'volume_momentum_{period}d'] = df['volume'] / df['volume'].shift(period) - 1
    
    # Regime Detection System
    # Volatility Regime
    df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
    vol_threshold_high = df['volatility_20d'].rolling(window=60).quantile(0.75)
    vol_threshold_low = df['volatility_20d'].rolling(window=60).quantile(0.25)
    
    df['volatility_regime'] = 'medium'
    df.loc[df['volatility_20d'] > vol_threshold_high, 'volatility_regime'] = 'high'
    df.loc[df['volatility_20d'] < vol_threshold_low, 'volatility_regime'] = 'low'
    
    # Trend Regime
    df['price_trend_slope'] = df['close'].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
    )
    trend_threshold = df['price_trend_slope'].abs().rolling(window=60).quantile(0.67)
    
    df['trend_regime'] = 'sideways'
    df.loc[df['price_trend_slope'] > trend_threshold, 'trend_regime'] = 'uptrend'
    df.loc[df['price_trend_slope'] < -trend_threshold, 'trend_regime'] = 'downtrend'
    
    # Regime-Adaptive Weighting
    def get_regime_weights(vol_regime, trend_regime):
        if vol_regime == 'high':
            price_weight, volume_weight = 0.3, 0.7
        elif vol_regime == 'low':
            price_weight, volume_weight = 0.7, 0.3
        else:  # medium
            price_weight, volume_weight = 0.5, 0.5
        
        if trend_regime == 'uptrend':
            continuation_weight, reversal_weight = 0.7, 0.3
        elif trend_regime == 'downtrend':
            continuation_weight, reversal_weight = 0.3, 0.7
        else:  # sideways
            continuation_weight, reversal_weight = 0.5, 0.5
            
        return price_weight, volume_weight, continuation_weight, reversal_weight
    
    # Divergence Pattern Analysis
    def calculate_divergence_signal(row, period):
        price_mom = row[f'price_momentum_{period}d']
        volume_mom = row[f'volume_momentum_{period}d']
        
        if pd.isna(price_mom) or pd.isna(volume_mom):
            return 0
        
        # Bullish divergence: price down but volume up (accumulation)
        if price_mom < 0 and volume_mom > 0:
            return 1
        # Bearish divergence: price up but volume down (distribution)
        elif price_mom > 0 and volume_mom < 0:
            return -1
        # Confirmation: both same direction
        elif price_mom * volume_mom > 0:
            return 0.5 if price_mom > 0 else -0.5
        else:
            return 0
    
    for period in [5, 10, 20]:
        df[f'divergence_{period}d'] = df.apply(
            lambda row: calculate_divergence_signal(row, period), axis=1
        )
    
    # Multi-timeframe consistency
    def get_timeframe_consistency(row):
        signals = [row['divergence_5d'], row['divergence_10d'], row['divergence_20d']]
        positive_count = sum(1 for s in signals if s > 0)
        negative_count = sum(1 for s in signals if s < 0)
        
        if positive_count == 3:
            return 1.0  # Strong bullish
        elif negative_count == 3:
            return -1.0  # Strong bearish
        elif positive_count == 2:
            return 0.7  # Moderate bullish
        elif negative_count == 2:
            return -0.7  # Moderate bearish
        elif positive_count == 1 and negative_count == 0:
            return 0.3  # Weak bullish
        elif negative_count == 1 and positive_count == 0:
            return -0.3  # Weak bearish
        else:
            return 0  # Contradictory
    
    df['timeframe_consistency'] = df.apply(get_timeframe_consistency, axis=1)
    
    # Volatility-Adjusted Signal Generation
    def generate_final_signal(row):
        if pd.isna(row['volatility_regime']) or pd.isna(row['trend_regime']):
            return 0
        
        price_weight, volume_weight, continuation_weight, reversal_weight = get_regime_weights(
            row['volatility_regime'], row['trend_regime']
        )
        
        # Calculate weighted momentum
        price_momentum_avg = np.mean([
            row['price_momentum_5d'], 
            row['price_momentum_10d'], 
            row['price_momentum_20d']
        ])
        
        volume_momentum_avg = np.mean([
            row['volume_momentum_5d'], 
            row['volume_momentum_10d'], 
            row['volume_momentum_20d']
        ])
        
        weighted_momentum = (price_momentum_avg * price_weight + 
                           volume_momentum_avg * volume_weight)
        
        # Apply regime-adaptive weighting to divergence signal
        base_signal = row['timeframe_consistency']
        
        # Adjust for trend regime
        if row['trend_regime'] == 'uptrend' and base_signal > 0:
            signal_strength = base_signal * continuation_weight
        elif row['trend_regime'] == 'uptrend' and base_signal < 0:
            signal_strength = base_signal * reversal_weight
        elif row['trend_regime'] == 'downtrend' and base_signal < 0:
            signal_strength = base_signal * continuation_weight
        elif row['trend_regime'] == 'downtrend' and base_signal > 0:
            signal_strength = base_signal * reversal_weight
        else:
            signal_strength = base_signal * 0.5
        
        # Combine with momentum and scale by volatility
        combined_signal = signal_strength * (1 + abs(weighted_momentum))
        
        # Volatility scaling (inverse relationship)
        if not pd.isna(row['volatility_20d']) and row['volatility_20d'] > 0:
            volatility_scale = 1 / (1 + row['volatility_20d'])
            final_signal = combined_signal * volatility_scale
        else:
            final_signal = combined_signal
        
        return final_signal
    
    df['alpha_factor'] = df.apply(generate_final_signal, axis=1)
    
    return df['alpha_factor']
