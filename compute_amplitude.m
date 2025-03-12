function Amplitude = compute_amplitude(signal)
    try
        % Find the max and min values in the signal
        max_val = max(signal); 
        min_val = min(signal); 
        
        % Select the dominant peak (preserve sign)
        if abs(max_val) > abs(min_val)
            Amplitude = max_val; % Keep original sign
        else
            Amplitude = min_val; % Keep original sign
        end

    catch
        warning('Error in compute_amplitude, returning NaN.');
        Amplitude = NaN;
    end
end
