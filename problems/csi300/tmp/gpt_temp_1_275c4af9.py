import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for safe division
    def safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    
    # Helper function for safe log
    def safe_log(x):
        return np.log(np.abs(x) + 1e-10) * np.sign(x)
    
    # Fractal Volatility Divergence
    # Micro
    hl_ratio_micro = safe_div(data['high'] - data['low'], 
                             data['high'].shift(1) - data['low'].shift(1))
    close_range_ratio_micro = safe_div(data['close'] - data['close'].shift(1), 
                                     data['high'] - data['low'])
    momentum_ratio_micro = safe_div(data['close'] - data['close'].shift(1), 
                                  data['close'].shift(1) - data['close'].shift(2))
    entropy_micro = -np.abs(momentum_ratio_micro) * safe_log(np.abs(momentum_ratio_micro))
    micro_vol = hl_ratio_micro * close_range_ratio_micro * entropy_micro
    
    # Meso
    hl_ratio_meso = safe_div(data['high'] - data['low'], 
                            data['high'].shift(5) - data['low'].shift(5))
    close_range_ratio_meso = safe_div(data['close'] - data['close'].shift(5), 
                                    data['high'] - data['low'])
    momentum_ratio_meso = safe_div(data['close'] - data['close'].shift(3), 
                                 data['close'].shift(3) - data['close'].shift(6))
    entropy_meso = -np.abs(momentum_ratio_meso) * safe_log(np.abs(momentum_ratio_meso))
    volume_ratio_meso = safe_div(data['volume'], data['volume'].shift(3))
    meso_vol = hl_ratio_meso * close_range_ratio_meso * entropy_meso * volume_ratio_meso
    
    # Macro
    hl_ratio_macro = safe_div(data['high'] - data['low'], 
                            data['high'].shift(20) - data['low'].shift(20))
    close_range_ratio_macro = safe_div(data['close'] - data['close'].shift(20), 
                                     data['high'] - data['low'])
    momentum_ratio_macro = safe_div(data['close'] - data['close'].shift(10), 
                                  data['close'].shift(10) - data['close'].shift(20))
    entropy_macro = -np.abs(momentum_ratio_macro) * safe_log(np.abs(momentum_ratio_macro))
    amount_ratio_macro = safe_div(data['amount'], data['amount'].shift(10))
    macro_vol = hl_ratio_macro * close_range_ratio_macro * entropy_macro * amount_ratio_macro
    
    # Entropic Momentum Divergence
    # Micro
    mom_ratio_micro = safe_div(data['close'] - data['close'].shift(1), 
                             data['close'].shift(1) - data['close'].shift(2))
    entropy_mom_micro = -np.abs(mom_ratio_micro) * safe_log(np.abs(mom_ratio_micro))
    position_change_micro = (safe_div(data['close'] - data['low'], data['high'] - data['low']) - 
                           safe_div(data['close'].shift(1) - data['low'].shift(1), 
                                  data['high'].shift(1) - data['low'].shift(1)))
    micro_mom = mom_ratio_micro * entropy_mom_micro * position_change_micro
    
    # Meso
    recent_std = data['close'].rolling(window=5).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    past_std = data['close'].shift(4).rolling(window=5).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    vol_ratio_meso = safe_div(recent_std, past_std)
    momentum_ratio_meso_mom = safe_div(data['close'] - data['close'].shift(3), 
                                     data['close'].shift(3) - data['close'].shift(6))
    entropy_meso_mom = -np.abs(momentum_ratio_meso_mom) * safe_log(np.abs(momentum_ratio_meso_mom))
    volume_ratio_meso_mom = safe_div(data['volume'], data['volume'].shift(3))
    meso_mom = vol_ratio_meso * entropy_meso_mom * volume_ratio_meso_mom
    
    # Macro
    recent_std_macro = data['close'].rolling(window=11).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    past_std_macro = data['close'].shift(10).rolling(window=11).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    vol_ratio_macro = safe_div(recent_std_macro, past_std_macro)
    momentum_ratio_macro_mom = safe_div(data['close'] - data['close'].shift(10), 
                                      data['close'].shift(10) - data['close'].shift(20))
    entropy_macro_mom = -np.abs(momentum_ratio_macro_mom) * safe_log(np.abs(momentum_ratio_macro_mom))
    amount_ratio_macro_mom = safe_div(data['amount'], data['amount'].shift(10))
    macro_mom = vol_ratio_macro * entropy_macro_mom * amount_ratio_macro_mom
    
    # Cross-Scale Fractal Divergence
    vol_divergence = np.abs(micro_vol - meso_vol) * np.abs(meso_vol - macro_vol)
    mom_divergence = np.abs(micro_mom - meso_mom) * np.abs(meso_mom - macro_mom)
    combined_fractal_divergence = vol_divergence * mom_divergence
    
    # Position-Volume Integration
    # Opening Fracture
    open_gap = safe_div(data['open'] - data['close'].shift(1), data['high'] - data['low'])
    gap_magnitude = np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    close_open_ratio = safe_div(data['close'] - data['open'], data['high'] - data['low'])
    entropy_open = -np.abs(close_open_ratio) * safe_log(np.abs(close_open_ratio))
    position_bias = safe_div(data['close'] - data['low'], data['high'] - data['low']) - 0.5
    opening_fracture = open_gap * gap_magnitude * entropy_open * position_bias
    
    # Midday Asymmetry
    mid_position = safe_div(data['close'] - (data['high'] + data['low']) / 2, data['high'] - data['low'])
    volume_change = safe_div(data['volume'], data['volume'].shift(1))
    mom_ratio_mid = safe_div(data['close'] - data['close'].shift(1), 
                           np.abs(data['close'].shift(1) - data['close'].shift(2)))
    entropy_mid = -np.abs(mom_ratio_mid) * safe_log(np.abs(mom_ratio_mid))
    volume_amount_ratio = safe_div(data['volume'], data['amount'])
    midday_asymmetry = mid_position * volume_change * entropy_mid * volume_amount_ratio
    
    # Volume Momentum
    price_change = data['close'] - data['close'].shift(1)
    vol_amount_ratio = safe_div(data['volume'], data['amount'])
    range_ratio = safe_div(data['high'] - data['low'], data['close'].shift(1))
    mom_ratio_vol = safe_div(data['close'] - data['close'].shift(1), 
                           data['close'].shift(1) - data['close'].shift(2))
    entropy_vol = -np.abs(mom_ratio_vol) * safe_log(np.abs(mom_ratio_vol))
    volume_momentum = price_change * vol_amount_ratio * range_ratio * entropy_vol
    
    position_volume_integration = (opening_fracture + midday_asymmetry) / 2
    
    # Fractal Divergence Regimes
    # Volatility Regime
    vol_change = np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1)))
    vol_regime = vol_change * np.sign(combined_fractal_divergence)
    
    # Momentum Regime
    mom_sign_current = np.sign(data['close'] - data['close'].shift(1))
    mom_sign_prev = np.sign(data['close'].shift(1) - data['close'].shift(2))
    mom_sign_prev2 = np.sign(data['close'].shift(2) - data['close'].shift(3))
    momentum_count = ((mom_sign_current == mom_sign_prev).astype(int) + 
                     (mom_sign_prev == mom_sign_prev2).astype(int))
    momentum_regime = momentum_count * np.sign(combined_fractal_divergence)
    
    # Volume Regime
    vol_sign = np.sign(data['volume'] - data['volume'].shift(1))
    price_sign = np.sign(data['close'] - data['close'].shift(1))
    volume_regime = (vol_sign == price_sign).astype(int) * np.sign(volume_momentum)
    
    # Position Regime
    position_bias_current = safe_div(data['close'] - data['low'], data['high'] - data['low']) - 0.5
    position_regime = position_bias_current * np.sign(position_volume_integration)
    
    regime_multipliers = (vol_regime + momentum_regime + volume_regime + position_regime) / 4
    
    # Final Alpha Assembly
    core_divergence = combined_fractal_divergence * (position_volume_integration + volume_momentum) / 2
    alpha = core_divergence * regime_multipliers * np.sign(core_divergence)
    
    return alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
