import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Multi-Timeframe Efficiency Analysis
    # Short-term efficiency: |Close(t) - Close(t-5)| / sum(TR from t-4 to t)
    short_term_eff = abs(close - close.shift(5)) / tr.rolling(window=5).sum()
    
    # Medium-term efficiency: |Close(t) - Close(t-10)| / sum(TR from t-9 to t)
    medium_term_eff = abs(close - close.shift(10)) / tr.rolling(window=10).sum()
    
    # Efficiency gradient: (10-day efficiency - 5-day efficiency) * sign(Close(t) - Close(t-5))
    eff_gradient = (medium_term_eff - short_term_eff) * np.sign(close - close.shift(5))
    
    # Volatility-Regime Classification
    atr_3 = tr.rolling(window=3).mean()
    atr_10 = tr.rolling(window=10).mean()
    vol_ratio = atr_3 / atr_10
    
    def get_vol_regime(ratio):
        if ratio > 1.2:
            return 'high'
        elif ratio < 0.8:
            return 'low'
        else:
            return 'normal'
    
    vol_regime = vol_ratio.apply(get_vol_regime)
    
    # Microstructure-Anchored Momentum
    volume = df['volume']
    anchored_momentum = (close - ((high * volume + low * volume) / (2 * volume))) / close
    
    # Volume Flow Asymmetry
    close_low_vol = (close - low) * volume
    high_close_vol = (high - close) * volume
    
    ratio_10 = (close_low_vol.rolling(window=10).mean() / 
                high_close_vol.rolling(window=10).mean())
    ratio_5 = (close_low_vol.rolling(window=5).mean() / 
               high_close_vol.rolling(window=5).mean())
    
    flow_divergence = (np.log(ratio_10) - np.log(ratio_5)) * np.sign(close - close.shift(5))
    
    # Price-Volume Co-Movement
    returns = close.pct_change()
    volume_change = volume.pct_change()
    
    def count_co_movement(window_data):
        ret, vol_chg = window_data
        matches = sum(1 for i in range(len(ret)) if np.sign(ret.iloc[i]) == np.sign(vol_chg.iloc[i]))
        return matches / len(ret) if len(ret) > 0 else 0
    
    co_movement_eff = pd.Series([
        count_co_movement((returns.iloc[i-4:i+1], volume_change.iloc[i-4:i+1])) 
        if i >= 4 else 0 
        for i in range(len(df))
    ], index=df.index)
    
    # Intraday Fractal Confirmation
    open_price = df['open']
    intraday_eff = (close - open_price) / (high - low)
    intraday_eff = intraday_eff.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Momentum Quality
    ret_3_day = close / close.shift(3) - 1
    ret_5_day = close / close.shift(5) - 1
    momentum_direction = np.sign(ret_3_day) * np.sign(ret_5_day)
    
    # Adaptive Signal Synthesis
    regime_adaptive_base = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        if i < 10:  # Need enough data for calculations
            continue
            
        if vol_regime.iloc[i] == 'high':
            regime_adaptive_base.iloc[i] = (eff_gradient.iloc[i] * 
                                           flow_divergence.iloc[i] * 
                                           anchored_momentum.iloc[i])
        elif vol_regime.iloc[i] == 'low':
            regime_adaptive_base.iloc[i] = (eff_gradient.iloc[i] * 
                                           co_movement_eff.iloc[i] * 
                                           anchored_momentum.iloc[i])
        else:  # normal
            regime_adaptive_base.iloc[i] = (eff_gradient.iloc[i] * 
                                           (short_term_eff.iloc[i] + medium_term_eff.iloc[i]) / 2 * 
                                           flow_divergence.iloc[i])
    
    # Final Alpha
    final_alpha = (regime_adaptive_base * momentum_direction * 
                   co_movement_eff * intraday_eff)
    
    return final_alpha.fillna(0)
