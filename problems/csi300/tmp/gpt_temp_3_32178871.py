import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Volume Confluence with Price Efficiency alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize the factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Very Short-Term Momentum (1-3 days)
    data['price_acceleration'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2)).replace(0, np.nan)
    data['volume_momentum_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Short-Term Momentum (5-10 days)
    data['momentum_persistence'] = data['close'].rolling(window=5).apply(
        lambda x: np.sum(x > x.shift(1).fillna(method='ffill')) / 5, raw=False
    )
    data['volume_expansion_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['price_range_efficiency'] = (data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    ).replace(0, np.nan)
    
    # Medium-Term Momentum (15-25 days)
    data['trend_quality'] = (data['close'] - data['close'].shift(15)) / (
        (data['close'] - data['close'].shift(1)).rolling(window=15).apply(lambda x: np.sum(np.abs(x)), raw=False)
    ).replace(0, np.nan)
    
    # Calculate rolling correlation for volume trend consistency
    def rolling_corr(x):
        if len(x) < 15:
            return np.nan
        prices = x[:, 0]
        volumes = x[:, 1]
        price_changes = prices[1:] - prices[:-1]
        return np.corrcoef(price_changes, volumes[1:])[0, 1] if len(price_changes) > 1 else np.nan
    
    price_volume_data = np.column_stack([data['close'].values, data['volume'].values])
    data['volume_trend_consistency'] = pd.Series(
        [rolling_corr(price_volume_data[max(0, i-14):i+1]) for i in range(len(data))],
        index=data.index
    )
    
    data['momentum_stability'] = 1 - (
        (data['close'] - data['close'].shift(1)).rolling(window=15).std() / 
        (data['close'] - data['close'].shift(15)).abs()
    ).replace([np.inf, -np.inf], np.nan)
    
    # Price Efficiency and Market Microstructure
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / (
        data['high'].shift(1) - data['low'].shift(1)
    ).replace(0, np.nan)
    
    data['auction_absorption'] = (data['close'] - data['open']) / (
        data['open'] - data['close'].shift(1)
    ).replace(0, np.nan)
    
    data['high_low_capture'] = (data['close'] - data['low']) / (
        data['high'] - data['low']
    ).replace(0, np.nan)
    
    data['upper_shadow_rejection'] = (
        data['high'] - np.maximum(data['open'], data['close'])
    ) / (data['high'] - data['low']).replace(0, np.nan)
    
    data['lower_shadow_support'] = (
        np.minimum(data['open'], data['close']) - data['low']
    ) / (data['high'] - data['low']).replace(0, np.nan)
    
    data['end_of_day_pressure'] = (data['close'] - data['close'].shift(1)) / (
        data['high'] - data['low']
    ).replace(0, np.nan)
    
    data['settlement_efficiency'] = np.abs(
        data['close'] - (data['high'] + data['low']) / 2
    ) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Distribution and Flow Analysis
    data['vwap_deviation'] = (data['close'] - (
        (data['high'] + data['low'] + data['close']) / 3
    )) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Timeframe Momentum Confluence Scoring
    def calculate_momentum_confluence(row):
        scores = []
        
        # Very Short-Term components
        if not pd.isna(row['price_acceleration']):
            scores.append(np.tanh(row['price_acceleration']))
        if not pd.isna(row['volume_momentum_alignment']):
            scores.append(row['volume_momentum_alignment'])
        if not pd.isna(row['intraday_efficiency']):
            scores.append(row['intraday_efficiency'])
        
        # Short-Term components
        if not pd.isna(row['momentum_persistence']):
            scores.append(2 * row['momentum_persistence'] - 1)
        if not pd.isna(row['volume_expansion_ratio']):
            scores.append(np.tanh(row['volume_expansion_ratio'] - 1))
        if not pd.isna(row['price_range_efficiency']):
            scores.append(row['price_range_efficiency'])
        
        # Medium-Term components
        if not pd.isna(row['trend_quality']):
            scores.append(np.tanh(row['trend_quality']))
        if not pd.isna(row['volume_trend_consistency']):
            scores.append(row['volume_trend_consistency'])
        if not pd.isna(row['momentum_stability']):
            scores.append(np.clip(row['momentum_stability'], -1, 1))
        
        return np.nanmean(scores) if scores else np.nan
    
    # Calculate momentum confluence for each row
    momentum_confluence = []
    for idx in data.index:
        row = data.loc[idx]
        momentum_confluence.append(calculate_momentum_confluence(row))
    
    data['momentum_confluence'] = momentum_confluence
    
    # Price Efficiency Scoring
    def calculate_price_efficiency(row):
        efficiency_scores = []
        
        if not pd.isna(row['opening_pressure']):
            efficiency_scores.append(-np.abs(row['opening_pressure']))  # Lower absolute pressure is better
        
        if not pd.isna(row['auction_absorption']):
            efficiency_scores.append(np.tanh(row['auction_absorption']))
        
        if not pd.isna(row['high_low_capture']):
            efficiency_scores.append(2 * row['high_low_capture'] - 1)  # Convert to -1 to 1 scale
        
        if not pd.isna(row['upper_shadow_rejection']):
            efficiency_scores.append(-row['upper_shadow_rejection'])  # Lower rejection is better
        
        if not pd.isna(row['lower_shadow_support']):
            efficiency_scores.append(row['lower_shadow_support'])  # Higher support is better
        
        if not pd.isna(row['end_of_day_pressure']):
            efficiency_scores.append(np.tanh(row['end_of_day_pressure']))
        
        if not pd.isna(row['settlement_efficiency']):
            efficiency_scores.append(-row['settlement_efficiency'])  # Lower deviation is better
        
        if not pd.isna(row['vwap_deviation']):
            efficiency_scores.append(-np.abs(row['vwap_deviation']))  # Lower absolute deviation is better
        
        return np.nanmean(efficiency_scores) if efficiency_scores else np.nan
    
    # Calculate price efficiency for each row
    price_efficiency = []
    for idx in data.index:
        row = data.loc[idx]
        price_efficiency.append(calculate_price_efficiency(row))
    
    data['price_efficiency'] = price_efficiency
    
    # Final Alpha Factor Generation
    def generate_final_factor(row):
        momentum_score = row['momentum_confluence'] if not pd.isna(row['momentum_confluence']) else 0
        efficiency_score = row['price_efficiency'] if not pd.isna(row['price_efficiency']) else 0
        
        # Volume confirmation multiplier
        volume_multiplier = 1.0
        if not pd.isna(row['volume_expansion_ratio']):
            volume_multiplier *= min(2.0, max(0.5, row['volume_expansion_ratio']))
        
        if not pd.isna(row['volume_trend_consistency']):
            volume_multiplier *= (1 + row['volume_trend_consistency'])
        
        # Combined factor with volume confirmation
        combined_factor = momentum_score * efficiency_score * volume_multiplier
        
        # Apply stability adjustment
        if not pd.isna(row['momentum_stability']):
            stability_adjustment = np.clip(row['momentum_stability'], 0.5, 1.5)
            combined_factor *= stability_adjustment
        
        return combined_factor
    
    # Generate final factor values
    final_factor = []
    for idx in data.index:
        row = data.loc[idx]
        final_factor.append(generate_final_factor(row))
    
    factor = pd.Series(final_factor, index=data.index)
    
    # Normalize the factor using rolling z-score (21-day window)
    factor_mean = factor.rolling(window=21, min_periods=10).mean()
    factor_std = factor.rolling(window=21, min_periods=10).std()
    factor_normalized = (factor - factor_mean) / factor_std.replace(0, np.nan)
    
    # Handle any remaining NaN values
    factor_normalized = factor_normalized.fillna(0)
    
    return factor_normalized
