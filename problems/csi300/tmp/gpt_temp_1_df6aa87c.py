import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Price EMA components
    ema_uf_p = pd.Series(index=df.index, dtype=float)
    ema_f_p = pd.Series(index=df.index, dtype=float)
    ema_m_p = pd.Series(index=df.index, dtype=float)
    ema_s_p = pd.Series(index=df.index, dtype=float)
    
    # Volume EMA components
    ema_uf_v = pd.Series(index=df.index, dtype=float)
    ema_f_v = pd.Series(index=df.index, dtype=float)
    ema_m_v = pd.Series(index=df.index, dtype=float)
    ema_s_v = pd.Series(index=df.index, dtype=float)
    
    # Volatility components
    ema_range = pd.Series(index=df.index, dtype=float)
    ema_vol_chg = pd.Series(index=df.index, dtype=float)
    
    # Initialize first values
    if len(df) > 0:
        ema_uf_p.iloc[0] = df['close'].iloc[0]
        ema_f_p.iloc[0] = df['close'].iloc[0]
        ema_m_p.iloc[0] = df['close'].iloc[0]
        ema_s_p.iloc[0] = df['close'].iloc[0]
        
        ema_uf_v.iloc[0] = df['volume'].iloc[0]
        ema_f_v.iloc[0] = df['volume'].iloc[0]
        ema_m_v.iloc[0] = df['volume'].iloc[0]
        ema_s_v.iloc[0] = df['volume'].iloc[0]
        
        ema_range.iloc[0] = df['high'].iloc[0] - df['low'].iloc[0]
        ema_vol_chg.iloc[0] = 0
    
    # Calculate EMA components
    for i in range(1, len(df)):
        # Price EMAs
        ema_uf_p.iloc[i] = 0.5 * df['close'].iloc[i] + 0.5 * ema_uf_p.iloc[i-1]
        ema_f_p.iloc[i] = 0.3 * df['close'].iloc[i] + 0.7 * ema_f_p.iloc[i-1]
        ema_m_p.iloc[i] = 0.15 * df['close'].iloc[i] + 0.85 * ema_m_p.iloc[i-1]
        ema_s_p.iloc[i] = 0.1 * df['close'].iloc[i] + 0.9 * ema_s_p.iloc[i-1]
        
        # Volume EMAs
        ema_uf_v.iloc[i] = 0.5 * df['volume'].iloc[i] + 0.5 * ema_uf_v.iloc[i-1]
        ema_f_v.iloc[i] = 0.3 * df['volume'].iloc[i] + 0.7 * ema_f_v.iloc[i-1]
        ema_m_v.iloc[i] = 0.15 * df['volume'].iloc[i] + 0.85 * ema_m_v.iloc[i-1]
        ema_s_v.iloc[i] = 0.1 * df['volume'].iloc[i] + 0.9 * ema_s_v.iloc[i-1]
        
        # Volatility EMAs
        daily_range = df['high'].iloc[i] - df['low'].iloc[i]
        ema_range.iloc[i] = 0.3 * daily_range + 0.7 * ema_range.iloc[i-1]
        
        vol_chg = abs(df['volume'].iloc[i] - df['volume'].iloc[i-1])
        ema_vol_chg.iloc[i] = 0.3 * vol_chg + 0.7 * ema_vol_chg.iloc[i-1]
    
    # Calculate scale factors
    price_scale = 1 / (ema_range + 0.0001)
    volume_scale = 1 / (ema_vol_chg + 0.0001)
    
    # Calculate momentum differences
    # Price momentum hierarchy
    ultra_fast_price_mom = (ema_uf_p - ema_f_p) * price_scale
    fast_price_mom = (ema_f_p - ema_m_p) * price_scale
    medium_price_mom = (ema_m_p - ema_s_p) * price_scale
    ultra_slow_price_mom = (ema_uf_p - ema_s_p) * price_scale
    
    # Volume momentum hierarchy
    ultra_fast_volume_mom = (ema_uf_v - ema_f_v) * volume_scale
    fast_volume_mom = (ema_f_v - ema_m_v) * volume_scale
    medium_volume_mom = (ema_m_v - ema_s_v) * volume_scale
    ultra_slow_volume_mom = (ema_uf_v - ema_s_v) * volume_scale
    
    # Convergence-divergence matrix
    ultra_fast_conv = ultra_fast_price_mom - ultra_fast_volume_mom
    fast_conv = fast_price_mom - fast_volume_mom
    medium_conv = medium_price_mom - medium_volume_mom
    ultra_slow_conv = ultra_slow_price_mom - ultra_slow_volume_mom
    
    # Dynamic regime adaptation
    volatility_ratio = ema_range / (ema_range.shift(7).fillna(ema_range) + 0.0001)
    trend_strength = abs(ema_uf_p - ema_s_p) / (ema_range + 0.0001)
    volume_regime = ema_vol_chg / (ema_vol_chg.shift(7).fillna(ema_vol_chg) + 0.0001)
    
    # Adaptive weight functions
    ultra_fast_weight = np.maximum(0, np.minimum(1, 2.5 - volatility_ratio - trend_strength))
    fast_weight = np.maximum(0, np.minimum(1, 2 - abs(1 - volatility_ratio) - abs(1 - volume_regime)))
    medium_weight = np.maximum(0, np.minimum(1, volatility_ratio + volume_regime - 1))
    ultra_slow_weight = np.maximum(0, np.minimum(1, trend_strength - 1))
    
    # Final alpha composition
    alpha = (ultra_fast_conv * ultra_fast_weight + 
             fast_conv * fast_weight + 
             medium_conv * medium_weight + 
             ultra_slow_conv * ultra_slow_weight)
    
    return alpha
