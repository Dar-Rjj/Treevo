import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1]
        prev_prev_data = df.iloc[i-2]
        
        # Extract current day values
        open_price = current_data['open']
        high = current_data['high']
        low = current_data['low']
        close = current_data['close']
        volume = current_data['volume']
        
        # Previous day values
        prev_open = prev_data['open']
        prev_high = prev_data['high']
        prev_low = prev_data['low']
        prev_close = prev_data['close']
        prev_volume = prev_data['volume']
        
        # Calculate daily range
        daily_range = high - low
        prev_range = prev_high - prev_low
        
        # Avoid division by zero
        if daily_range == 0:
            daily_range = 1e-10
        if prev_range == 0:
            prev_range = 1e-10
        
        # 1. Bidirectional Price Impact Asymmetry
        # Directional price impact efficiency
        up_efficiency = 0.0
        down_efficiency = 0.0
        
        if close > open_price:
            up_efficiency = (high - open_price) / daily_range
        if close < open_price:
            down_efficiency = (open_price - low) / daily_range
        
        # Asymmetric momentum persistence
        up_momentum_durability = 0
        down_momentum_fragility = 0
        
        # Check consecutive up days with high efficiency
        if (close > open_price and prev_close > prev_open and 
            up_efficiency > 0.6 and (prev_high - prev_open) / prev_range > 0.6):
            up_momentum_durability = 1
        
        # Check consecutive down days with low efficiency
        if (close < open_price and prev_close < prev_open and 
            down_efficiency < 0.4 and (prev_open - prev_low) / prev_range < 0.4):
            down_momentum_fragility = 1
        
        # Directional pressure imbalance
        bullish_pressure = ((close - low) / daily_range) * volume if daily_range > 0 else 0
        bearish_pressure = ((high - close) / daily_range) * volume if daily_range > 0 else 0
        pressure_imbalance = bullish_pressure - bearish_pressure
        
        # 2. Liquidity Absorption & Exhaustion Cycles
        # Volume absorption patterns
        price_movement = abs(close - open_price)
        movement_ratio = price_movement / daily_range if daily_range > 0 else 0
        
        high_volume_absorption = 0
        if volume > prev_volume * 1.2 and movement_ratio < 0.3:
            high_volume_absorption = 1
        
        low_volume_exhaustion = 0
        if volume < prev_volume * 0.8 and movement_ratio > 0.7:
            low_volume_exhaustion = 1
        
        # Volume acceleration
        volume_acceleration = (volume - prev_volume) / prev_volume if prev_volume > 0 else 0
        
        # Volume persistence
        volume_persistence = 0
        if (volume > prev_volume and prev_volume > prev_prev_data['volume']):
            volume_persistence = 1
        
        # 3. Microstructure Momentum Quality
        # Opening gap momentum
        gap = open_price - prev_close
        gap_momentum = gap / prev_close if prev_close > 0 else 0
        
        # Midday momentum acceleration
        midday_range = high - low
        midday_momentum = (close - open_price) / open_price if open_price > 0 else 0
        
        # Closing momentum strength
        closing_strength = (close - low) / daily_range if daily_range > 0 else 0
        
        # Momentum fragmentation
        price_reversal_frequency = 0
        if (close > open_price and prev_close < prev_open) or (close < open_price and prev_close > prev_open):
            price_reversal_frequency = 1
        
        # 4. Asymmetric Volatility Response
        # Directional volatility
        up_volatility = daily_range if close > open_price else 0
        down_volatility = daily_range if close < open_price else 0
        
        # Volatility persistence
        volatility_clustering = 0
        if (close > open_price and prev_close > prev_open and 
            daily_range > prev_range * 0.8):
            volatility_clustering = 1
        
        volatility_dispersion = 0
        if (close < open_price and prev_close < prev_open and 
            daily_range < prev_range * 1.2):
            volatility_dispersion = 1
        
        # 5. Adaptive Signal Generation Framework
        # Combine signals with weights
        signal = 0.0
        
        # High absorption + asymmetric up-efficiency → strong bullish momentum
        if high_volume_absorption and up_efficiency > 0.7:
            signal += 2.0
        
        # Low exhaustion + asymmetric down-efficiency → weak bearish momentum
        if low_volume_exhaustion and down_efficiency > 0.6:
            signal -= 1.5
        
        # Momentum fragmentation + volatility asymmetry → trend reversal signal
        if price_reversal_frequency and abs(up_volatility - down_volatility) > daily_range * 0.3:
            signal -= 1.0
        
        # Liquidity flow acceleration + microstructure quality → momentum continuation
        if volume_acceleration > 0.1 and closing_strength > 0.6:
            signal += 1.5
        
        # Asymmetric pressure imbalance + volatility response → directional conviction
        if abs(pressure_imbalance) > volume * 0.1 and volatility_clustering:
            if pressure_imbalance > 0:
                signal += 1.2
            else:
                signal -= 1.2
        
        # Additional momentum quality factors
        if up_momentum_durability:
            signal += 0.8
        if down_momentum_fragility:
            signal -= 0.8
        
        if volume_persistence and midday_momentum > 0.01:
            signal += 0.5
        
        # Normalize by volatility to reduce noise
        volatility_adjustment = daily_range / open_price if open_price > 0 else 1.0
        if volatility_adjustment > 0:
            signal = signal / volatility_adjustment
        
        factor_values.iloc[i] = signal
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
