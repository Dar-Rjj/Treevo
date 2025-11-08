import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the input dataframe to avoid modifying the original
    data = df.copy()
    
    # Initialize the output series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate returns for volatility calculations
    returns = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Calculate moving averages and other rolling statistics
    data['ma_close_3'] = data['close'].rolling(window=3).mean()
    data['ma_close_5'] = data['close'].rolling(window=5).mean()
    data['ma_close_10'] = data['close'].rolling(window=10).mean()
    data['ma_close_20'] = data['close'].rolling(window=20).mean()
    data['ma_volume_5'] = data['volume'].rolling(window=5).mean()
    data['ma_volume_20'] = data['volume'].rolling(window=20).mean()
    data['ma_amount_5'] = data['amount'].rolling(window=5).mean()
    
    # Calculate volatility measures
    short_term_vol = returns.rolling(window=5).std()
    medium_term_vol = returns.rolling(window=20).std()
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ma_true_range = true_range.rolling(window=10).mean()
    range_expansion = true_range / ma_true_range
    
    # Calculate price-volume correlation
    price_change = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    pv_corr = price_change.rolling(window=5).corr(data['volume'])
    
    for i in range(1, len(data)):
        if i < 20:  # Skip initial period for reliable calculations
            continue
            
        current_date = data.index[i]
        
        # 1. Intraday Momentum Efficiency Factor
        intraday_return = (data.loc[current_date, 'close'] - data.loc[current_date, 'open']) / data.loc[current_date, 'open']
        overnight_gap = (data.loc[current_date, 'open'] - data.loc[data.index[i-1], 'close']) / data.loc[data.index[i-1], 'close']
        price_range_efficiency = (data.loc[current_date, 'high'] - data.loc[current_date, 'low']) / data.loc[current_date, 'open']
        
        momentum_consistency = np.sign(intraday_return) * np.sign(overnight_gap)
        efficiency_ratio = abs(intraday_return) / price_range_efficiency if price_range_efficiency > 0 else 0
        volume_confirmation = data.loc[current_date, 'volume'] / data.loc[data.index[i-1], 'volume'] if data.loc[data.index[i-1], 'volume'] > 0 else 1
        
        primary_momentum = intraday_return * efficiency_ratio
        timing_adjustment = primary_momentum * momentum_consistency
        
        if volume_confirmation > 1.2:
            ime_signal = timing_adjustment * volume_confirmation
        else:
            ime_signal = 0
        
        # 2. Volatility Regime Adaptive Factor
        if volatility_ratio.loc[current_date] > 1.5:
            # High volatility regime - mean reversion
            price_deviation = (data.loc[current_date, 'close'] - data.loc[current_date, 'ma_close_10']) / data.loc[current_date, 'ma_close_10']
            vraf_signal = -price_deviation * volatility_ratio.loc[current_date]
        elif volatility_ratio.loc[current_date] < 0.7:
            # Low volatility regime - momentum
            price_acceleration = returns.loc[current_date] - returns.loc[data.index[i-1]]
            vraf_signal = price_acceleration * (1 / volatility_ratio.loc[current_date])
        else:
            # Normal regime - hybrid
            momentum_component = returns.loc[current_date] * (data.loc[current_date, 'volume'] / data.loc[data.index[i-1], 'volume'])
            reversal_component = -0.5 * ((data.loc[current_date, 'close'] - data.loc[current_date, 'ma_close_10']) / data.loc[current_date, 'ma_close_10'])
            vraf_signal = momentum_component + reversal_component
        
        # 3. Liquidity-Driven Price Pressure Factor
        amount_based_flow = data.loc[current_date, 'amount'] * np.sign(data.loc[current_date, 'close'] - data.loc[current_date, 'open'])
        volume_based_flow = data.loc[current_date, 'volume'] * (data.loc[current_date, 'close'] - data.loc[current_date, 'open']) / data.loc[current_date, 'open']
        flow_consistency = np.sign(amount_based_flow) * np.sign(volume_based_flow)
        
        effective_spread = (data.loc[current_date, 'high'] - data.loc[current_date, 'low']) / ((data.loc[current_date, 'high'] + data.loc[current_date, 'low']) / 2)
        volume_concentration = data.loc[current_date, 'volume'] / data.loc[current_date, 'ma_volume_5']
        price_impact = abs(data.loc[current_date, 'close'] - data.loc[current_date, 'open']) / data.loc[current_date, 'open']
        
        raw_pressure = (amount_based_flow + volume_based_flow) / 2
        liquidity_adjustment = raw_pressure / effective_spread if effective_spread > 0 else 0
        
        if flow_consistency == 1 and volume_concentration > 1:
            ldpf_signal = liquidity_adjustment * 1.5
        elif flow_consistency == 1:
            ldpf_signal = liquidity_adjustment
        else:
            ldpf_signal = liquidity_adjustment * 0.5
        
        # 4. Range Breakout Validation Factor
        breakout_direction = np.sign(data.loc[current_date, 'close'] - data.loc[data.index[i-1], 'close'])
        volume_surge = data.loc[current_date, 'volume'] / data.loc[current_date, 'ma_volume_20']
        price_level = data.loc[current_date, 'close'] / data.loc[current_date, 'ma_close_20']
        
        base_breakout = range_expansion.loc[current_date] * breakout_direction
        volume_validation = base_breakout * volume_surge
        
        if volume_surge > 1.5 and pv_corr.loc[current_date] > 0.3:
            rbf_signal = volume_validation * 2
        elif volume_surge > 1.2:
            rbf_signal = volume_validation * 1.2
        else:
            rbf_signal = volume_validation * 0.8
        
        # 5. Multi-Timeframe Momentum Convergence
        very_short_momentum = returns.loc[current_date]
        short_momentum = (data.loc[current_date, 'close'] - data.loc[current_date, 'ma_close_3']) / data.loc[current_date, 'ma_close_3']
        medium_momentum = (data.loc[current_date, 'close'] - data.loc[current_date, 'ma_close_5']) / data.loc[current_date, 'ma_close_5']
        
        momentum_signals = [very_short_momentum, short_momentum, medium_momentum]
        momentum_alignment = sum(1 for m in momentum_signals if m > 0)
        momentum_magnitude = np.mean([abs(m) for m in momentum_signals])
        volume_trend = data.loc[current_date, 'volume'] / data.loc[current_date, 'ma_volume_5']
        price_level_support = data.loc[current_date, 'close'] / data.loc[current_date, 'ma_close_10']
        
        base_convergence = momentum_alignment * momentum_magnitude
        volume_enhancement = base_convergence * volume_trend
        
        if price_level_support > 1.02:
            mtmc_signal = volume_enhancement * 1.3
        elif price_level_support >= 0.98:
            mtmc_signal = volume_enhancement
        else:
            mtmc_signal = volume_enhancement * 0.7
        
        # 6. Amount-Volume Divergence Factor
        amount_per_volume = data.loc[current_date, 'amount'] / data.loc[current_date, 'volume'] if data.loc[current_date, 'volume'] > 0 else 0
        amount_per_volume_prev = data.loc[data.index[i-1], 'amount'] / data.loc[data.index[i-1], 'volume'] if data.loc[data.index[i-1], 'volume'] > 0 else 0
        
        amount_volume_ratio_change = amount_per_volume / amount_per_volume_prev if amount_per_volume_prev > 0 else 1
        amount_trend = data.loc[current_date, 'amount'] / data.loc[current_date, 'ma_amount_5']
        volume_trend_av = data.loc[current_date, 'volume'] / data.loc[current_date, 'ma_volume_5']
        
        trend_divergence = np.sign(amount_trend - 1) * np.sign(volume_trend_av - 1)
        price_confirmation = np.sign(data.loc[current_date, 'close'] - data.loc[data.index[i-1], 'close'])
        
        raw_divergence = amount_volume_ratio_change * trend_divergence
        price_alignment = raw_divergence * price_confirmation
        
        if abs(raw_divergence) > 1.2 and price_alignment > 0:
            avdf_signal = price_alignment * 2
        elif abs(raw_divergence) > 1.1:
            avdf_signal = price_alignment * 1.5
        else:
            avdf_signal = price_alignment
        
        # Combine all factors with equal weights
        combined_signal = (ime_signal + vraf_signal + ldpf_signal + rbf_signal + mtmc_signal + avdf_signal) / 6
        
        factor_values.loc[current_date] = combined_signal
    
    # Fill NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
