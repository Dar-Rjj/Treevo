import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Analysis
    df = df.copy()
    
    # Calculate momentum components
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_accel'] = df['momentum_5d'] - df['momentum_10d']
    
    # Momentum convergence score
    df['momentum_convergence'] = np.sign(df['momentum_5d']) * np.sign(df['momentum_10d'])
    
    # Volume-Price Dynamics Assessment
    df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_adjusted_range'] = (df['high'] - df['low']) * df['volume']
    df['volume_price_efficiency'] = (df['close'] - df['close'].shift(1)) / (df['volume'] + 1e-8)
    
    # Volume confirmation
    df['volume_confirmation'] = np.where(
        (df['momentum_5d'] > 0) & (df['volume_momentum'] > 0), 1,
        np.where((df['momentum_5d'] < 0) & (df['volume_momentum'] < 0), -1, 0)
    )
    
    # Volatility Regime Classification
    df['returns'] = df['close'].pct_change()
    df['volatility_20d'] = df['returns'].rolling(window=20).std()
    
    # Average True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_10d'] = df['true_range'].rolling(window=10).mean()
    
    # Volatility regime
    vol_median = df['volatility_20d'].rolling(window=60).median()
    atr_median = df['atr_10d'].rolling(window=60).median()
    
    df['vol_regime'] = np.where(
        (df['volatility_20d'] > vol_median) & (df['atr_10d'] > atr_median), 
        'high', 
        'low'
    )
    
    # Convergence-Divergence Pattern Detection
    # Momentum-Volume Alignment
    df['momentum_volume_alignment'] = np.where(
        np.sign(df['momentum_accel']) == np.sign(df['volume_momentum']), 1, -1
    )
    
    df['volume_efficiency_alignment'] = np.where(
        np.sign(df['momentum_5d']) == np.sign(df['volume_price_efficiency']), 1, -1
    )
    
    # Range expansion/contraction
    df['range_momentum'] = df['volume_adjusted_range'] / df['volume_adjusted_range'].shift(5) - 1
    df['range_alignment'] = np.where(
        np.sign(df['momentum_5d']) == np.sign(df['range_momentum']), 1, -1
    )
    
    # Multi-timeframe consistency
    df['momentum_direction_alignment'] = np.where(
        np.sign(df['momentum_5d']) == np.sign(df['momentum_10d']), 1, 0
    )
    
    df['convergence_strength'] = abs(df['momentum_5d']) * abs(df['momentum_10d']) * df['momentum_convergence']
    
    # Regime-Adaptive Signal Generation
    signal = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.isna(df.iloc[i]['vol_regime']):
            signal.iloc[i] = 0
            continue
            
        # Base signal components
        momentum_signal = df.iloc[i]['momentum_convergence'] * df.iloc[i]['convergence_strength']
        volume_signal = df.iloc[i]['volume_confirmation']
        alignment_signal = (df.iloc[i]['momentum_volume_alignment'] + 
                          df.iloc[i]['volume_efficiency_alignment'] + 
                          df.iloc[i]['range_alignment']) / 3
        
        if df.iloc[i]['vol_regime'] == 'high':
            # High volatility regime signals
            if (df.iloc[i]['momentum_convergence'] > 0 and df.iloc[i]['volume_confirmation'] > 0):
                # Strong momentum continuation
                regime_signal = momentum_signal * 1.2
            elif (df.iloc[i]['momentum_accel'] > 0 and df.iloc[i]['range_momentum'] > 0):
                # Volatility breakout
                regime_signal = alignment_signal * df.iloc[i]['momentum_accel'] * 1.5
            elif (df.iloc[i]['momentum_convergence'] < 0 and df.iloc[i]['volume_confirmation'] < 0):
                # Reversal warning
                regime_signal = momentum_signal * volume_signal * 0.8
            else:
                regime_signal = (momentum_signal + volume_signal + alignment_signal) / 3
                
        else:  # Low volatility regime
            if (df.iloc[i]['momentum_direction_alignment'] > 0 and 
                abs(df.iloc[i]['volume_momentum']) < 0.1):
                # Trend persistence
                regime_signal = momentum_signal * 0.8
            elif (df.iloc[i]['momentum_accel'] > 0 and df.iloc[i]['volume_momentum'] > 0.2):
                # Breakout anticipation
                regime_signal = alignment_signal * df.iloc[i]['momentum_accel'] * 1.3
            elif (abs(df.iloc[i]['momentum_5d']) > 0.05 and df.iloc[i]['volume_momentum'] < -0.1):
                # Mean reversion
                regime_signal = -momentum_signal * 0.7
            else:
                regime_signal = (momentum_signal + volume_signal + alignment_signal) / 3
        
        # Signal strength calibration
        vol_weight = df.iloc[i]['volatility_20d'] / df.iloc[i]['volatility_20d'].rolling(window=60).mean()
        convergence_weight = abs(df.iloc[i]['convergence_strength'])
        volume_weight = abs(df.iloc[i]['volume_confirmation'])
        
        final_signal = regime_signal * vol_weight * convergence_weight * (1 + volume_weight * 0.5)
        signal.iloc[i] = final_signal
    
    return signal.fillna(0)
