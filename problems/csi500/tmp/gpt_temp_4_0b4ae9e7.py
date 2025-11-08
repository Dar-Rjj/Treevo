import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal-Microstructure Momentum Alpha
    Combines multi-scale fractal momentum with microstructure anchoring signals
    """
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Scale Fractal Momentum
    # Short-term (1-5 days) price path complexity using Hurst exponent approximation
    def hurst_exponent(series, window):
        """Approximate Hurst exponent using rescaled range analysis"""
        lags = range(2, min(window, 10))
        tau = []
        for lag in lags:
            rs_values = []
            for i in range(len(series) - window + 1):
                window_data = series.iloc[i:i+window]
                # Calculate R/S statistic
                mean = window_data.mean()
                deviations = window_data - mean
                Z = deviations.cumsum()
                R = Z.max() - Z.min()
                S = window_data.std()
                if S > 0:
                    rs_values.append(R / S)
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) > 1:
            lags_log = np.log(lags[:len(tau)])
            hurst = np.polyfit(lags_log, tau, 1)[0]
            return hurst
        return 0.5
    
    # Calculate multi-scale Hurst exponents
    hurst_short = pd.Series(index=data.index, dtype=float)
    hurst_medium = pd.Series(index=data.index, dtype=float)
    hurst_long = pd.Series(index=data.index, dtype=float)
    
    for i in range(59, len(data)):
        if i >= 4:
            hurst_short.iloc[i] = hurst_exponent(data['close'].iloc[i-4:i+1], 5)
        if i >= 19:
            hurst_medium.iloc[i] = hurst_exponent(data['close'].iloc[i-19:i+1], 20)
        if i >= 59:
            hurst_long.iloc[i] = hurst_exponent(data['close'].iloc[i-59:i+1], 60)
    
    # Multi-scale momentum signals
    mom_short = data['close'] / data['close'].shift(5) - 1
    mom_medium = data['close'] / data['close'].shift(20) - 1
    mom_long = data['close'] / data['close'].shift(60) - 1
    
    # Fractal momentum score (higher for persistent trends)
    fractal_momentum = (hurst_medium * mom_medium).fillna(0) + \
                      (hurst_long * mom_long).fillna(0) - \
                      (hurst_short * np.abs(mom_short)).fillna(0)
    
    # Microstructure Anchoring
    # Opening price relative to previous structure
    prev_close = data['close'].shift(1)
    open_gap = (data['open'] - prev_close) / prev_close
    
    # Intraday high-volume price levels (using amount as proxy for dollar volume)
    volume_weighted_price = (data['high'] + data['low'] + data['close']) / 3
    vwap_5d = (volume_weighted_price * data['amount']).rolling(5).sum() / data['amount'].rolling(5).sum()
    
    # Price relative to microstructure anchors
    price_vs_vwap = data['close'] / vwap_5d - 1
    
    # Overnight gap behavior (gap persistence)
    gap_persistence = (open_gap.rolling(5).mean() * np.sign(open_gap)).fillna(0)
    
    # Microstructure anchoring score
    microstructure_anchor = (price_vs_vwap * 0.4 + gap_persistence * 0.3 + 
                           (data['close'] / data['open'] - 1) * 0.3).fillna(0)
    
    # Fractal-Microstructure Convergence
    # Volume concentration at fractal boundaries
    high_vol_days = data['volume'] > data['volume'].rolling(20).quantile(0.7)
    fractal_boundary_strength = (data['high'].rolling(5).max() - data['low'].rolling(5).min()) / data['close']
    volume_convergence = (high_vol_days.astype(float) * fractal_boundary_strength).fillna(0)
    
    # Price rejection at microstructure levels
    upper_rejection = ((data['high'] - data['close']) / (data['high'] - data['low'])).fillna(0.5)
    lower_rejection = ((data['close'] - data['low']) / (data['high'] - data['low'])).fillna(0.5)
    rejection_signal = (upper_rejection - lower_rejection).rolling(3).mean().fillna(0)
    
    # Momentum strength near key anchors
    anchor_strength = (fractal_momentum.rolling(5).std() * 
                      microstructure_anchor.rolling(5).std()).fillna(0)
    
    convergence_score = (volume_convergence * 0.4 + rejection_signal * 0.3 + 
                        anchor_strength * 0.3).fillna(0)
    
    # Integrated Alpha Factors
    # Fractal momentum with volume confirmation
    volume_confirmation = (fractal_momentum * 
                          (data['volume'] / data['volume'].rolling(20).mean())).fillna(0)
    
    # Microstructure breakout with fractal validation
    breakout_signal = ((data['close'] > data['high'].rolling(5).max()) | 
                      (data['close'] < data['low'].rolling(5).min())).astype(float)
    fractal_validation = breakout_signal * hurst_medium.fillna(0.5)
    
    # Multi-scale momentum alignment
    momentum_alignment = (np.sign(mom_short) * np.sign(mom_medium) * np.sign(mom_long) * 
                         (np.abs(mom_short) + np.abs(mom_medium) + np.abs(mom_long)) / 3).fillna(0)
    
    # Final alpha factor combining all components
    alpha = (volume_confirmation * 0.3 + 
             fractal_validation * 0.25 + 
             momentum_alignment * 0.2 + 
             convergence_score * 0.15 + 
             microstructure_anchor * 0.1)
    
    return alpha.fillna(0)
