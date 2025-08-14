def heuristics_v2(data):
    # Calculate Daily Price Change
    data['daily_price_change'] = data['close'] - data['close'].shift(1)
    
    # Calculate Price Momentum
    data['price_momentum'] = data['daily_price_change'].rolling(window=5).sum()
    
    # Calculate Average Volume
    data['avg_volume'] = data['volume'].rolling(window=5).mean()
    
    # Apply Volume Filter
    def volume_filter(row):
        if row['volume'] > 1.5 * row['avg_volume']:
            return row['price_momentum']
        else:
            return 0
    
    data['alpha_factor'] = data.apply(volume_filter, axis=1)
    
    return data['alpha_factor']
