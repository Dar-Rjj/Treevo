import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic components
    data['close_ret_2'] = data['close'] / data['close'].shift(2) - 1
    data['range_today'] = data['high'] - data['low']
    data['range_yesterday'] = data['range_today'].shift(1)
    data['range_ratio'] = data['range_today'] / data['range_yesterday']
    data['gap'] = np.abs(data['open'] - data['close'].shift(1))
    data['intraday_move'] = data['close'] - data['open']
    data['gap_absorption'] = data['intraday_move'] / data['gap']
    data['range_efficiency'] = data['intraday_move'] / data['range_today']
    
    # Asymmetric Volatility Resonance
    # Directional Volatility Resonance
    upside_vol = []
    downside_vol = []
    for i in range(len(data)):
        if i < 4:
            upside_vol.append(np.nan)
            downside_vol.append(np.nan)
            continue
            
        window = data['close'].iloc[i-4:i+1]
        upside_returns = []
        downside_returns = []
        
        for j in range(1, len(window)):
            if window.iloc[j] > window.iloc[j-1]:
                upside_returns.append(window.iloc[j])
            elif window.iloc[j] < window.iloc[j-1]:
                downside_returns.append(window.iloc[j])
        
        upside_vol.append(np.std(upside_returns) if len(upside_returns) > 1 else 0)
        downside_vol.append(np.std(downside_returns) if len(downside_returns) > 1 else 0)
    
    data['upside_resonance'] = np.array(upside_vol) * data['close_ret_2'] * data['range_ratio']
    data['downside_resonance'] = np.array(downside_vol) * data['close_ret_2'] * data['range_ratio']
    data['asymmetric_ratio'] = data['upside_resonance'] / data['downside_resonance']
    
    # Gap-Induced Resonance
    data['gap_resonance_magnitude'] = data['gap'] * data['close_ret_2'] * data['range_ratio']
    data['gap_absorption_resonance'] = data['gap_absorption'] * data['close_ret_2']
    data['gap_volatility_resonance'] = data['gap_resonance_magnitude'] * data['gap_absorption_resonance']
    
    # Intraday Resonance Structure
    data['volatility_compression'] = data[['open', 'high', 'low', 'close']].std(axis=1) / data['range_today']
    data['volatility_compression_resonance'] = data['volatility_compression'] * data['close_ret_2']
    data['range_efficiency_resonance'] = data['range_efficiency'] * data['close_ret_2'] * data['range_ratio']
    data['intraday_resonance_divergence'] = data['range_efficiency_resonance'] - data['volatility_compression_resonance']
    
    # Volume-Frequency Asymmetry
    # Note: Since we don't have intraday volume data, we'll use daily volume as proxy
    data['volume_price_resonance'] = data['volume'] * (data['close'] - data['close'].shift(1)) / data['amount'] * data['close_ret_2']
    
    # Divergence Resonance Persistence
    divergence_persistence = []
    for i in range(len(data)):
        if i < 4:
            divergence_persistence.append(np.nan)
            continue
            
        total = 0
        for j in range(5):
            vol_diff = np.sign(data['volume'].iloc[i-j] - data['volume'].iloc[i-j-1])
            price_diff = np.sign(data['close'].iloc[i-j] - data['close'].iloc[i-j-1])
            total += vol_diff * price_diff
        
        divergence_persistence.append(total * data['close_ret_2'].iloc[i])
    
    data['divergence_resonance_persistence'] = np.array(divergence_persistence)
    data['resonance_mismatch_amplitude'] = np.abs(data['volume_price_resonance']) * data['range_today'] * data['close_ret_2']
    
    # Trade Size Resonance
    data['large_trade_resonance'] = np.where(
        data['close'] > data['open'],
        data['amount'] / data['volume'] * data['close_ret_2'] * data['range_ratio'],
        0
    )
    data['small_trade_resonance'] = np.where(
        data['close'] < data['open'],
        data['volume'] / data['amount'] * data['close_ret_2'] * data['range_ratio'],
        0
    )
    data['size_resonance_bias'] = data['large_trade_resonance'] / data['small_trade_resonance'].replace(0, np.nan)
    
    # Regime-Sensitive Resonance Momentum
    data['range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['high_vol_regime'] = (data['range_today'] > 2 * data['range_5d']) & (
        data['close_ret_2'] * data['range_ratio'] > data['close'].pct_change(5) * data['range_today'].shift(2) / data['range_today'].shift(4)
    )
    data['low_vol_regime'] = (data['range_today'] < 0.5 * data['range_5d']) & (
        data['close_ret_2'] * data['range_ratio'] <= data['close'].pct_change(5) * data['range_today'].shift(2) / data['range_today'].shift(4)
    )
    
    data['high_vol_resonance_momentum'] = (data['close'] - data['close'].shift(1)) / data['range_today'] * data['close_ret_2'] * data['range_ratio']
    data['low_vol_resonance_momentum'] = (data['close'] - data['close'].shift(5)) / data['range_today'] * data['close_ret_2'] * data['range_ratio']
    data['high_volume_resonance_momentum'] = (data['close'] - data['close'].shift(1)) * data['volume'] / data['amount'] * data['close_ret_2'] * data['range_ratio']
    
    # Asymmetric Convergence Resonance
    data['vv_resonance_alignment'] = data['asymmetric_ratio'] * data['size_resonance_bias']
    data['gap_volume_resonance'] = data['gap_absorption_resonance'] * data['volume_price_resonance']
    data['compression_concentration_resonance'] = data['volatility_compression_resonance'] * data['volume_price_resonance']
    
    data['skew_resonance_momentum'] = data['asymmetric_ratio'] * data['high_vol_resonance_momentum']
    data['divergence_resonance_momentum'] = data['divergence_resonance_persistence'] * data['low_vol_resonance_momentum']
    data['size_resonance_momentum'] = data['size_resonance_bias'] * data['high_volume_resonance_momentum']
    
    data['high_vol_convergence_resonance'] = data['vv_resonance_alignment'] * data['skew_resonance_momentum']
    data['low_vol_convergence_resonance'] = data['compression_concentration_resonance'] * data['divergence_resonance_momentum']
    data['high_volume_convergence_resonance'] = data['gap_volume_resonance'] * data['size_resonance_momentum']
    
    # Adaptive Resonance Factor Construction
    data['volatility_resonance_weight'] = data['range_today'] / data['range_5d'] * data['close_ret_2']
    data['volume_resonance_weight'] = data['volume'] / data['volume'].shift(5) * data['close_ret_2']
    data['regime_resonance_confidence'] = np.abs(data['volatility_resonance_weight'] - 1) * np.abs(data['volume_resonance_weight'] - 1) * data['close_ret_2']
    
    # Multi-Regime Component Selection
    volatility_driven_resonance = np.where(
        data['high_vol_regime'],
        data['high_vol_convergence_resonance'] * data['volatility_resonance_weight'],
        data['low_vol_convergence_resonance'] * data['volatility_resonance_weight']
    )
    
    volume_driven_resonance = data['high_volume_convergence_resonance'] * data['volume_resonance_weight']
    
    convergence_resonance = (
        np.where(data['high_vol_regime'], data['skew_resonance_momentum'], 0) +
        np.where(data['low_vol_regime'], data['divergence_resonance_momentum'], 0)
    ) * data['regime_resonance_confidence']
    
    asymmetry_resonance = (
        np.where(data['high_vol_regime'], data['gap_volatility_resonance'], 0) +
        np.where(data['volume'] > data['volume'].shift(5), data['volume_price_resonance'], 0)
    ) * data['size_resonance_bias']
    
    # Final Alpha Synthesis
    core_resonance_alpha = volatility_driven_resonance + volume_driven_resonance
    convergence_enhanced = core_resonance_alpha * (1 + convergence_resonance)
    asymmetry_adjusted = core_resonance_alpha * (1 + asymmetry_resonance)
    
    final_alpha = core_resonance_alpha * data['intraday_resonance_divergence'] * data['size_resonance_bias']
    
    return final_alpha
