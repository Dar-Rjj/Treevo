import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Pressure-Efficiency Divergence Factor
    """
    data = df.copy()
    
    # Volatility Regime Detection
    # Calculate 5-day Average True Range (Short-term volatility)
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['short_term_vol'] = data['TR'].rolling(window=5, min_periods=3).mean()
    
    # Calculate 20-day Price Standard Deviation (Long-term volatility)
    data['long_term_vol'] = data['close'].rolling(window=20, min_periods=10).std()
    
    # Determine Volatility Regime
    data['high_vol_regime'] = data['short_term_vol'] > data['long_term_vol']
    
    # Regime-Adaptive Efficiency Calculation
    # Price Path Efficiency Component
    def calculate_price_efficiency(row, window, current_idx):
        if current_idx < window:
            return np.nan
        
        net_movement = row['close'] - data.iloc[current_idx - window]['close']
        total_path = 0
        
        for i in range(window + 1):
            idx = current_idx - window + i
            if idx >= 0:
                total_path += abs(data.iloc[idx]['high'] - data.iloc[idx]['low'])
        
        if total_path == 0:
            return 0
        return net_movement / total_path
    
    # Calculate regime-specific price efficiencies
    price_efficiencies = []
    for i in range(len(data)):
        if data.iloc[i]['high_vol_regime']:
            # High volatility: 3-day efficiency
            eff = calculate_price_efficiency(data.iloc[i], 2, i)
        else:
            # Low volatility: 8-day efficiency
            eff = calculate_price_efficiency(data.iloc[i], 7, i)
        price_efficiencies.append(eff)
    
    data['price_efficiency'] = price_efficiencies
    
    # Volume Efficiency Component
    def calculate_volume_efficiency(row, window, current_idx):
        if current_idx < window:
            return np.nan
        
        total_volume = 0
        directional_volume = 0
        
        for i in range(window + 1):
            idx = current_idx - window + i
            if idx >= 0:
                daily_return = data.iloc[idx]['close'] / data.iloc[idx]['open'] - 1
                total_volume += data.iloc[idx]['volume']
                
                # Strong directional move: absolute return > 1%
                if abs(daily_return) > 0.01:
                    directional_volume += data.iloc[idx]['volume']
        
        if total_volume == 0:
            return 0
        return directional_volume / total_volume
    
    # Calculate regime-specific volume efficiencies
    volume_efficiencies = []
    for i in range(len(data)):
        if data.iloc[i]['high_vol_regime']:
            # High volatility: 3-day volume concentration
            eff = calculate_volume_efficiency(data.iloc[i], 2, i)
        else:
            # Low volatility: 8-day volume concentration
            eff = calculate_volume_efficiency(data.iloc[i], 7, i)
        volume_efficiencies.append(eff)
    
    data['volume_efficiency'] = volume_efficiencies
    
    # Intraday Pressure Accumulation
    # Buying Pressure
    data['buying_pressure'] = (
        ((data['close'] - data['open']) / data['open']).clip(lower=0) +
        ((data['high'] - data['close'].shift(1)) / data['close'].shift(1)).clip(lower=0)
    ) * data['volume']
    
    # Selling Pressure
    data['selling_pressure'] = (
        ((data['open'] - data['close']) / data['open']).clip(lower=0) +
        ((data['close'].shift(1) - data['low']) / data['close'].shift(1)).clip(lower=0)
    ) * data['volume']
    
    # Net Pressure Signal with 3-day EMA
    data['raw_pressure'] = data['buying_pressure'] - data['selling_pressure']
    data['net_pressure'] = data['raw_pressure'].ewm(span=3, adjust=False).mean()
    
    # Efficiency-Pressure Divergence Analysis
    divergence_strength = np.zeros(len(data))
    
    for i in range(len(data)):
        price_eff = data.iloc[i]['price_efficiency']
        net_press = data.iloc[i]['net_pressure']
        
        if pd.notna(price_eff) and pd.notna(net_press):
            if price_eff > 0 and net_press < 0:
                # Positive Efficiency / Negative Pressure Divergence
                divergence_strength[i] = abs(price_eff) * abs(net_press)
            elif price_eff < 0 and net_press > 0:
                # Negative Efficiency / Positive Pressure Divergence
                divergence_strength[i] = abs(price_eff) * abs(net_press)
    
    data['divergence_strength'] = divergence_strength
    
    # Composite Factor Construction
    # Volume-Confirmed Efficiency
    data['efficiency_confidence'] = data['price_efficiency'] * data['volume_efficiency']
    
    # Pressure-Efficiency Divergence Enhancement
    data['enhanced_signal'] = data['efficiency_confidence'] * (1 + data['divergence_strength'])
    
    # Regime-Adaptive Weighting
    regime_multiplier = np.where(data['high_vol_regime'], 0.7, 1.3)
    
    # Final Alpha Output
    alpha_factor = data['enhanced_signal'] * regime_multiplier
    
    return alpha_factor
