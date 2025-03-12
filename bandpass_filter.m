function filtered_signal = bandpass_filter(signal, time)
    fs = 1 / mean(diff(time)); % Compute sampling frequency
    f_low = 2000;  % Lower cutoff frequency (was 500)
    f_high = 30000; % Upper cutoff frequency (was 10000)



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

    % Design Butterworth bandpass filter
    [b, a] = butter(4, [f_low, f_high] / (fs / 2), 'bandpass');
    %[b, a] = cheby1(4, 0.5, [f_low, f_high] / (fs / 2), 'bandpass');
    
    % Ensure filter coefficients are valid
    if any(isnan(b)) || any(isnan(a))
        warning('Invalid filter coefficients, returning original signal.');
        filtered_signal = signal;
        return;
    end

    % Apply Bandpass Filter Safely
    try
        filtered_signal = filtfilt(b, a, signal);
    catch
        warning('Filtering failed due to unstable signal. Returning original signal.');
        filtered_signal = signal;
    end
end
