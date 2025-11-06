import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility regime identification
    # Short-term volatility (5-day)
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_vol'] = data['ret'].rolling(window=5).std()
    
    # Range volatility
    data['range_vol'] = (data['high'] - data['low']) / data['close']
    
    # Gap volatility
    data['gap_vol'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Regime classification
    data['vol_regime'] = 'normal'
    data.loc[(data['short_term_vol'] > 0.02) & (data['range_vol'] > 0.03), 'vol_regime'] = 'high'
    data.loc[(data['short_term_vol'] < 0.01) & (data['range_vol'] < 0.015), 'vol_regime'] = 'low'
    
    # Core momentum components
    data['price_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    data['amount_momentum'] = (data['amount'] - data['amount'].shift(5)) / data['amount'].shift(5)
    
    # Liquidity assessment
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['liquidity_score'] = (data['volume'] / data['volume_ma_5']) / data['range_vol']
    
    # Liquidity regime
    data['liq_regime'] = 'moderate'
    data.loc[data['liquidity_score'] > 2.0, 'liq_regime'] = 'strong'
    data.loc[data['liquidity_score'] < 1.0, 'liq_regime'] = 'poor'
    
    # Regime-adaptive signal processing
    signals = []
    
    for i in range(len(data)):
        if i < 8:  # Need enough data for calculations
            signals.append(0)
            continue
            
        row = data.iloc[i]
        
        if row['vol_regime'] == 'high':
            # High volatility regime
            core_signal = -1 * row['price_momentum'] / (1 + row['range_vol'])
            volume_adj = 1 + row['volume_momentum'] * abs(row['price_momentum'])
            signal = core_signal * volume_adj
            
        elif row['vol_regime'] == 'low':
            # Low volatility regime
            sign1 = np.sign(data.iloc[i]['close'] - data.iloc[i-1]['close'])
            sign2 = np.sign(data.iloc[i-1]['close'] - data.iloc[i-2]['close'])
            core_signal = row['price_momentum'] * sign1 * sign2
            amount_adj = 1 + (row['amount_momentum'] + row['volume_momentum']) / 2
            signal = core_signal * amount_adj
            
        else:
            # Normal volatility regime
            mom3 = data.iloc[i]['close'] / data.iloc[i-3]['close'] - 1
            mom8 = data.iloc[i]['close'] / data.iloc[i-8]['close'] - 1
            core_signal = 0.6 * mom3 + 0.4 * mom8
            signal = core_signal
        
        # Apply liquidity multiplier
        if row['liq_regime'] == 'strong':
            signal *= 1.2
        elif row['liq_regime'] == 'poor':
            signal *= 0.8
        
        # Multi-timeframe validation
        short_align = np.sign(data.iloc[i]['close'] - data.iloc[i-1]['close']) * np.sign(data.iloc[i-1]['close'] - data.iloc[i-2]['close'])
        med_consistency = row['price_momentum'] - (data.iloc[i-1]['close'] - data.iloc[i-4]['close']) / data.iloc[i-4]['close']
        
        # Count aligned momentum signs
        aligned_count = 0
        if short_align > 0:
            aligned_count += 1
        if med_consistency > 0:
            aligned_count += 1
        if row['price_momentum'] > 0:
            aligned_count += 1
            
        signal_strength = aligned_count * 0.2
        
        # Final signal with validation boost
        final_signal = signal * (1 + signal_strength)
        
        # Risk adjustment
        if row['short_term_vol'] > 0:
            final_signal /= row['short_term_vol']
            
        signals.append(final_signal)
    
    data['alpha'] = signals
    
    return data['alpha']
