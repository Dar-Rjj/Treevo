import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining momentum, liquidity, and regime signals.
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate basic price changes and ratios
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['amount_ratio'] = data['amount'] / data['amount'].shift(1)
    data['range_t'] = data['high'] - data['low']
    data['range_ratio'] = data['range_t'] / data['range_t'].shift(1)
    
    # Volume-Adjusted Momentum
    data['vol_adj_momentum'] = data['close_ret'] / data['volume_ratio']
    
    # Amount-Weighted Momentum
    data['amt_weighted_momentum'] = data['close_ret'] * data['amount_ratio']
    
    # Liquidity-Momentum Divergence
    data['liq_mom_div'] = (data['volume_ratio'] - 1) / data['close_ret'] - (data['amount_ratio'] - 1) / data['close_ret']
    
    # Volatility-Adjusted Momentum
    data['vol_adj_momentum_2'] = data['close_ret'] * data['range_ratio']
    
    # Liquidity-Regime Momentum
    data['liq_regime_momentum'] = data['close_ret'] * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Regime Transition Signal
    data['regime_transition'] = (np.sign(data['range_t'] - data['range_t'].shift(1)) * 
                                np.sign(data['volume'] - data['volume'].shift(1)))
    
    # Opening Momentum Efficiency
    data['open_mom_eff'] = (data['open'] - data['close'].shift(1)) / data['range_t'].shift(1)
    
    # Intraday Momentum Capture
    data['intraday_mom_capture'] = (data['close'] - data['open']) / data['range_t']
    
    # Range-Momentum Product
    data['range_mom_product'] = data['open_mom_eff'] * data['intraday_mom_capture']
    
    # Directional Asymmetry
    data['directional_asymmetry'] = np.where(data['close_ret'] > 0, data['close_ret'],
                                           np.where(data['close_ret'] < 0, data['close_ret'], 0))
    
    # Volume Response Gap
    data['volume_response_gap'] = np.where(data['volume'] > data['volume'].shift(1), data['vol_adj_momentum'],
                                         np.where(data['volume'] < data['volume'].shift(1), data['vol_adj_momentum'], 0))
    
    # Calculate consistency counts for multi-timeframe integration
    def calculate_consistency(window, shift_period=1):
        consistency = pd.Series(0, index=data.index)
        for i in range(len(data) - window + 1):
            if i >= shift_period:
                count = 0
                for j in range(i, i + window):
                    if j >= shift_period and j < len(data) - shift_period:
                        current_sign = np.sign(data['close'].iloc[j] / data['close'].iloc[j-shift_period] - 1)
                        prev_sign = np.sign(data['close'].iloc[j-1] / data['close'].iloc[j-1-shift_period] - 1)
                        if current_sign == prev_sign:
                            count += 1
                consistency.iloc[i + window - 1] = count / window
        return consistency
    
    # Short-Term Enhanced
    data['short_term_consistency'] = calculate_consistency(2, 1)
    data['short_term_enhanced'] = data['close_ret'] * data['short_term_consistency']
    
    # Medium-Term Enhanced
    data['medium_term_consistency'] = calculate_consistency(5, 5)
    data['medium_term_enhanced'] = (data['close'] / data['close'].shift(5) - 1) * data['medium_term_consistency']
    
    # Scale Alignment Factor
    data['scale_alignment'] = np.sign(data['short_term_enhanced']) * np.sign(data['medium_term_enhanced'])
    
    # Core Momentum
    data['core_momentum'] = data['close_ret'] * data['short_term_consistency']
    
    # Liquidity Momentum
    data['liquidity_momentum'] = (data['vol_adj_momentum'] * 
                                 np.sign(data['close_ret']) * 
                                 np.sign(data['volume'] - data['volume'].shift(1)) * 
                                 np.abs(data['close_ret']))
    
    # Regime Momentum
    data['regime_consistency'] = calculate_consistency(3, 1)
    data['regime_momentum'] = data['regime_transition'] * data['regime_consistency']
    
    # Final Composite Alpha
    data['final_alpha'] = data['core_momentum'] * data['liquidity_momentum'] * data['regime_momentum']
    
    # Handle any infinite or NaN values
    data['final_alpha'] = data['final_alpha'].replace([np.inf, -np.inf], np.nan)
    
    return data['final_alpha']
