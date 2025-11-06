import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Multi-Scale Reversal factor
    Combines multi-timeframe reversal detection with dynamic volatility regime classification
    and volume-price dynamics for robust return prediction
    """
    data = df.copy()
    
    # Dynamic Volatility Regime Classification
    # True Range calculation
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = abs(data['high'] - data['close'].shift(1))
    data['TR3'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # ATR for volatility measurement (5-day window)
    data['ATR'] = data['TR'].rolling(window=5).mean()
    
    # Volatility regime classification
    data['volatility_regime'] = pd.cut(data['ATR'] / data['close'], 
                                      bins=[0, 0.01, 0.03, float('inf')], 
                                      labels=['low', 'medium', 'high'])
    
    # Multi-Timeframe Reversal Detection
    # Short-term reversal (1-3 days)
    data['ret_1d'] = data['close'].pct_change(1)
    data['ret_2d'] = data['close'].pct_change(2)
    data['ret_3d'] = data['close'].pct_change(3)
    
    # Medium-term mean reversion (5-10 days)
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_10d'] = data['close'].pct_change(10)
    
    # Reversal acceleration analysis
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['acceleration'] = data['momentum_3d'] - data['momentum_5d']
    
    # Volume-Price Dynamics Integration
    # Volume trend (5-day moving average)
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_trend'] = data['volume'] / data['volume_ma5'] - 1
    
    # Volume anomaly detection (z-score based)
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    
    # Volume-price divergence
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['volume_price_corr'] = data['price_change'].rolling(window=5).corr(data['volume'])
    
    # Adaptive Signal Construction
    # Volatility-weighted reversal components
    regime_weights = {
        'low': {'short_term': 0.6, 'medium_term': 0.3, 'acceleration': 0.1},
        'medium': {'short_term': 0.4, 'medium_term': 0.4, 'acceleration': 0.2},
        'high': {'short_term': 0.3, 'medium_term': 0.3, 'acceleration': 0.4}
    }
    
    # Calculate regime-adaptive reversal signals
    reversal_signals = []
    for idx, row in data.iterrows():
        regime = row['volatility_regime']
        if pd.isna(regime):
            reversal_signals.append(np.nan)
            continue
            
        weights = regime_weights[regime]
        
        # Short-term reversal component (negative momentum)
        short_term = -0.5 * row['ret_1d'] - 0.3 * row['ret_2d'] - 0.2 * row['ret_3d']
        
        # Medium-term mean reversion component
        medium_term = -0.6 * row['ret_5d'] - 0.4 * row['ret_10d']
        
        # Acceleration component (reversal momentum)
        acceleration_comp = -row['acceleration']
        
        # Weighted combination
        regime_signal = (weights['short_term'] * short_term + 
                        weights['medium_term'] * medium_term + 
                        weights['acceleration'] * acceleration_comp)
        
        reversal_signals.append(regime_signal)
    
    data['reversal_signal'] = reversal_signals
    
    # Volume validation and timing
    volume_confirmation = []
    for idx, row in data.iterrows():
        if pd.isna(row['volume_trend']) or pd.isna(row['volume_zscore']) or pd.isna(row['volume_price_corr']):
            volume_confirmation.append(np.nan)
            continue
            
        # Volume confirmation score
        vol_trend_score = 1 if row['volume_trend'] > 0.1 else 0.5 if row['volume_trend'] > 0 else 0
        vol_anomaly_score = 1 if abs(row['volume_zscore']) > 1 else 0.5
        vol_price_score = 1 if row['volume_price_corr'] < -0.3 else 0.5 if row['volume_price_corr'] < 0 else 0
        
        volume_score = (vol_trend_score + vol_anomaly_score + vol_price_score) / 3
        volume_confirmation.append(volume_score)
    
    data['volume_confirmation'] = volume_confirmation
    
    # Final factor construction
    # Combine reversal signal with volume confirmation
    data['factor'] = data['reversal_signal'] * data['volume_confirmation']
    
    # Normalize by volatility regime
    volatility_scaling = []
    for idx, row in data.iterrows():
        regime = row['volatility_regime']
        if pd.isna(regime) or pd.isna(row['ATR']):
            volatility_scaling.append(np.nan)
            continue
            
        # Scale factor by inverse volatility for better risk-adjusted performance
        if regime == 'low':
            scale = 1.0 / (row['ATR'] / row['close'] + 0.005)
        elif regime == 'medium':
            scale = 1.0 / (row['ATR'] / row['close'] + 0.01)
        else:  # high volatility
            scale = 1.0 / (row['ATR'] / row['close'] + 0.02)
        
        volatility_scaling.append(scale)
    
    data['volatility_scaling'] = volatility_scaling
    data['final_factor'] = data['factor'] * data['volatility_scaling']
    
    # Clean up intermediate columns
    result = data['final_factor'].copy()
    
    return result
