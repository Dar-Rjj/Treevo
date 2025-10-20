import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Efficiency-Momentum Acceleration with Volume-Amount Regime Transition Divergence
    """
    data = df.copy()
    
    # Multi-Timeframe Efficiency-Momentum Framework
    # Efficiency-based momentum calculation
    data['return_5d'] = data['close'].pct_change(5)
    data['return_20d'] = data['close'].pct_change(20)
    data['return_60d'] = data['close'].pct_change(60)
    
    data['abs_return_5d'] = data['close'].pct_change().rolling(5).apply(lambda x: np.sum(np.abs(x)), raw=True)
    data['abs_return_20d'] = data['close'].pct_change().rolling(20).apply(lambda x: np.sum(np.abs(x)), raw=True)
    data['abs_return_60d'] = data['close'].pct_change().rolling(60).apply(lambda x: np.sum(np.abs(x)), raw=True)
    
    # Efficiency momentum calculations
    data['eff_mom_5d'] = data['return_5d'] / (data['abs_return_5d'] + 1e-8)
    data['eff_mom_20d'] = data['return_20d'] / (data['abs_return_20d'] + 1e-8)
    data['eff_mom_60d'] = data['return_60d'] / (data['abs_return_60d'] + 1e-8)
    
    # Efficiency acceleration signals
    data['eff_accel_primary'] = data['eff_mom_5d'] - data['eff_mom_20d']
    data['eff_accel_secondary'] = data['eff_mom_20d'] - data['eff_mom_60d']
    
    # Efficiency trend persistence
    data['eff_sign_5d'] = np.sign(data['eff_mom_5d'])
    data['eff_sign_20d'] = np.sign(data['eff_mom_20d'])
    data['eff_sign_60d'] = np.sign(data['eff_mom_60d'])
    
    data['eff_persistence'] = 0
    for i in range(1, len(data)):
        if (data['eff_sign_5d'].iloc[i] == data['eff_sign_5d'].iloc[i-1] and 
            data['eff_sign_20d'].iloc[i] == data['eff_sign_20d'].iloc[i-1]):
            data.loc[data.index[i], 'eff_persistence'] = data['eff_persistence'].iloc[i-1] + 1
    
    # Cross-timeframe alignment
    data['eff_align_sm'] = data['eff_sign_5d'] * data['eff_sign_20d']
    data['eff_align_ml'] = data['eff_sign_20d'] * data['eff_sign_60d']
    data['eff_convergence_strength'] = (data['eff_align_sm'] + data['eff_align_ml']) / 2
    
    # Volume-Amount Efficiency Divergence Dynamics
    # Volume efficiency patterns
    data['daily_return'] = data['close'].pct_change()
    data['vol_eff'] = data['daily_return'] / (data['volume'] + 1e-8)
    data['vol_eff_5d'] = data['vol_eff'].rolling(5).mean()
    data['vol_eff_20d'] = data['vol_eff'].rolling(20).mean()
    data['vol_eff_accel'] = data['vol_eff_5d'] - data['vol_eff_20d']
    data['vol_mom'] = data['volume'] / data['volume'].rolling(5).mean()
    
    # Amount-based efficiency signals
    data['amt_eff'] = data['daily_return'] / (data['amount'] + 1e-8)
    data['amt_eff_5d'] = data['amt_eff'].rolling(5).mean()
    data['amt_eff_20d'] = data['amt_eff'].rolling(20).mean()
    data['amt_eff_accel'] = data['amt_eff_5d'] - data['amt_eff_20d']
    data['amt_vol_align'] = np.sign(data['amt_eff_accel']) * np.sign(data['vol_eff_accel'])
    
    # Efficiency divergence detection
    data['price_vol_div'] = np.sign(data['eff_accel_primary']) * np.sign(data['vol_eff_accel'])
    data['price_amt_div'] = np.sign(data['eff_accel_primary']) * np.sign(data['amt_eff_accel'])
    data['multi_eff_div'] = (data['price_vol_div'] + data['price_amt_div']) / 2
    
    # Range-Based Efficiency Momentum Integration
    # Range momentum analysis
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_change_5d'] = data['daily_range'].pct_change(5)
    data['range_mom_10d'] = data['daily_range'] / data['daily_range'].rolling(10).mean() - 1
    data['range_trend_21d'] = data['daily_range'].rolling(21).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Range efficiency patterns
    data['range_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['range_compression'] = data['daily_range'].rolling(5).mean() / data['daily_range'].rolling(20).mean()
    data['range_expansion'] = (data['daily_range'] > data['daily_range'].rolling(5).mean()).astype(int)
    
    # Range-volume interaction
    data['range_vol_interaction'] = data['range_expansion'] * data['vol_eff_accel']
    
    # Efficiency Regime Transition Detection
    # Efficiency regime classification
    data['eff_regime'] = np.where(data['eff_mom_5d'] > data['eff_mom_20d'].rolling(20).mean(), 1, -1)
    data['eff_regime_prev'] = data['eff_regime'].shift(1)
    data['regime_transition'] = (data['eff_regime'] != data['eff_regime_prev']).astype(int)
    
    # Regime transition momentum
    data['regime_persistence'] = 0
    for i in range(1, len(data)):
        if data['eff_regime'].iloc[i] == data['eff_regime'].iloc[i-1]:
            data.loc[data.index[i], 'regime_persistence'] = data['regime_persistence'].iloc[i-1] + 1
    
    # Regime divergence patterns
    data['vol_eff_regime'] = np.where(data['vol_eff_5d'] > data['vol_eff_20d'].rolling(20).mean(), 1, -1)
    data['regime_div_price_vol'] = data['eff_regime'] * data['vol_eff_regime']
    
    # Divergence-Transition Integration Framework
    # Efficiency-volume divergence timing
    data['pos_div'] = ((data['eff_accel_primary'] > 0) & (data['vol_eff_accel'] < 0)).astype(int)
    data['neg_div'] = ((data['eff_accel_primary'] < 0) & (data['vol_eff_accel'] > 0)).astype(int)
    data['div_magnitude'] = np.abs(data['eff_accel_primary']) / (np.abs(data['vol_eff_accel']) + 1e-8)
    
    # Divergence persistence
    data['div_persistence'] = 0
    for i in range(1, len(data)):
        if (data['pos_div'].iloc[i] == data['pos_div'].iloc[i-1] or 
            data['neg_div'].iloc[i] == data['neg_div'].iloc[i-1]):
            data.loc[data.index[i], 'div_persistence'] = data['div_persistence'].iloc[i-1] + 1
    
    # Efficiency-amount flow divergence
    data['amt_flow_align'] = np.sign(data['eff_accel_primary']) * np.sign(data['amt_eff_accel'])
    data['amt_div_magnitude'] = np.abs(data['eff_accel_primary']) / (np.abs(data['amt_eff_accel']) + 1e-8)
    
    # Efficiency Persistence Analysis
    data['eff_improvement'] = (data['eff_mom_5d'] > data['eff_mom_5d'].shift(1)).astype(int)
    data['eff_streak'] = 0
    for i in range(1, len(data)):
        if data['eff_improvement'].iloc[i] == 1:
            data.loc[data.index[i], 'eff_streak'] = data['eff_streak'].iloc[i-1] + 1
    
    # Adaptive Divergence-Transition Signal Synthesis
    # Regime-divergence weighting scheme
    data['regime_weight'] = 1.0
    data.loc[(data['eff_regime'] == 1) & (data['range_expansion'] == 1), 'regime_weight'] = 1.5
    data.loc[(data['eff_regime'] == -1) & (data['range_compression'] < 1), 'regime_weight'] = 0.7
    data.loc[data['regime_transition'] == 1, 'regime_weight'] = 1.2
    data.loc[data['regime_div_price_vol'] == -1, 'regime_weight'] = 0.5
    
    # Divergence confirmation filters
    data['div_strength'] = 0
    strong_div_cond = (np.abs(data['multi_eff_div']) > 0.5) & (data['div_persistence'] > 2)
    weak_div_cond = (np.abs(data['multi_eff_div']) > 0) & (data['div_persistence'] <= 2)
    data.loc[strong_div_cond, 'div_strength'] = 1.5
    data.loc[weak_div_cond, 'div_strength'] = 0.8
    data.loc[~strong_div_cond & ~weak_div_cond, 'div_strength'] = 1.0
    
    # Composite factor construction
    # Base signal: regime-weighted efficiency momentum acceleration
    base_signal = data['eff_accel_primary'] * data['regime_weight']
    
    # Divergence multiplier: efficiency misalignment strength
    div_multiplier = 1 + (data['multi_eff_div'] * data['div_strength'] * 0.3)
    
    # Persistence enhancement: streak-based signal amplification
    persistence_enhance = 1 + (np.tanh(data['eff_streak'] / 10) * 0.2)
    
    # Transition premium: regime change detection bonus
    transition_premium = 1 + (data['regime_transition'] * 0.15)
    
    # Final alpha factor
    alpha_factor = (base_signal * div_multiplier * persistence_enhance * transition_premium + 
                   data['eff_convergence_strength'] * 0.1 + 
                   data['range_vol_interaction'] * 0.05 + 
                   data['amt_flow_align'] * 0.08)
    
    return alpha_factor
