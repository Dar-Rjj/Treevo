import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Reversal with Volume-Price Fractal Coherence factor
    Detects reversal signals based on fractal patterns and volume-price synchronization
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_required = ['open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in cols_required):
        missing = [col for col in cols_required if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate typical price and volume features
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume_ma'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    for i in range(4, len(df)):
        current_data = df.iloc[max(0, i-19):i+1]  # 20-day lookback including current
        
        if len(current_data) < 5:
            result.iloc[i] = 0
            continue
            
        # 1. Fractal Pattern Analysis
        fractal_scores = []
        volume_sync_scores = []
        
        # Analyze 5-point patterns for fractals
        for j in range(2, len(current_data)-2):
            window = current_data.iloc[j-2:j+3]
            highs = window['high'].values
            lows = window['low'].values
            volumes = window['volume'].values
            
            # Detect bearish fractal (higher highs pattern)
            if (highs[0] < highs[2] and highs[1] < highs[2] and 
                highs[3] < highs[2] and highs[4] < highs[2]):
                fractal_scores.append(-1)  # Bearish fractal
                # Volume synchronization check
                if volumes[2] > np.mean(volumes[[0,1,3,4]]) * 1.2:
                    volume_sync_scores.append(-1)
                else:
                    volume_sync_scores.append(0)
            
            # Detect bullish fractal (lower lows pattern)
            elif (lows[0] > lows[2] and lows[1] > lows[2] and 
                  lows[3] > lows[2] and lows[4] > lows[2]):
                fractal_scores.append(1)  # Bullish fractal
                # Volume synchronization check
                if volumes[2] > np.mean(volumes[[0,1,3,4]]) * 1.2:
                    volume_sync_scores.append(1)
                else:
                    volume_sync_scores.append(0)
        
        # 2. Fractal Density and Strength
        if fractal_scores:
            fractal_density = len(fractal_scores) / len(current_data)
            fractal_strength = np.mean(fractal_scores) if fractal_scores else 0
            volume_sync = np.mean(volume_sync_scores) if volume_sync_scores else 0
        else:
            fractal_density = 0
            fractal_strength = 0
            volume_sync = 0
        
        # 3. Momentum Divergence Detection
        current_price = current_data['close'].iloc[-1]
        price_5d_ago = current_data['close'].iloc[-6] if len(current_data) >= 6 else current_data['close'].iloc[0]
        momentum = (current_price - price_5d_ago) / price_5d_ago
        
        # Recent volatility for momentum normalization
        recent_volatility = current_data['high'].iloc[-5:].std() / current_data['close'].iloc[-5:].mean()
        normalized_momentum = momentum / (recent_volatility + 1e-6)
        
        # 4. Volume-Price Coherence Scoring
        current_volume_ratio = current_data['volume_ratio'].iloc[-1]
        
        # Coherence: volume should confirm price movement
        if abs(normalized_momentum) > 1.0:  # Strong momentum
            if (normalized_momentum > 0 and current_volume_ratio < 0.8) or \
               (normalized_momentum < 0 and current_volume_ratio < 0.8):
                coherence_score = -1  # Divergence - momentum weakening
            elif (normalized_momentum > 0 and current_volume_ratio > 1.2) or \
                 (normalized_momentum < 0 and current_volume_ratio > 1.2):
                coherence_score = 1  # Confirmation - momentum strengthening
            else:
                coherence_score = 0
        else:
            coherence_score = 0
        
        # 5. Alpha Factor Generation
        # Combine fractal strength, volume synchronization, and coherence
        fractal_component = fractal_strength * fractal_density * 2.0
        volume_component = volume_sync * 1.5
        coherence_component = coherence_score * 1.0
        
        # Momentum reversal probability weighting
        momentum_reversal_prob = np.tanh(abs(normalized_momentum) * 2) * np.sign(-normalized_momentum)
        
        # Final factor calculation
        alpha_value = (
            fractal_component + 
            volume_component + 
            coherence_component
        ) * (1 + momentum_reversal_prob)
        
        # Apply smoothing and bounds
        alpha_value = np.clip(alpha_value, -3, 3)
        
        result.iloc[i] = alpha_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
