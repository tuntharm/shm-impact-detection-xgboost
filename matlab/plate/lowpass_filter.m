function filtered_signal = lowpass_filter(signal, time)
    fs = 1 / mean(diff(time)); % Compute sampling frequency
    f_high = 30000; % Upper cutoff frequency (keep all below this)

    % Handle NaN or Inf values
    if any(isnan(signal)) || any(isinf(signal))
        warning('NaN or Inf detected in signal. Replacing with mean value.');
        signal(isnan(signal) | isinf(signal)) = mean(signal, 'omitnan');
    end

    % Skip filtering if signal is too small
    if std(signal) < 1e-10
        warning('Signal is almost constant, skipping filtering.');
        filtered_signal = signal; % Return original signal
        return;
    end

    % Design Butterworth lowpass filter
    [b, a] = butter(4, f_high / (fs / 2), 'low');
    
    % Ensure filter coefficients are valid
    if any(isnan(b)) || any(isnan(a))
        warning('Invalid filter coefficients, returning original signal.');
        filtered_signal = signal;
        return;
    end

    % Apply Lowpass Filter Safely
    try
        filtered_signal = filtfilt(b, a, signal);
    catch
        warning('Filtering failed due to unstable signal. Returning original signal.');
        filtered_signal = signal;
    end
end
