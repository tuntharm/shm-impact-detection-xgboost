function ToA = compute_ToA(signal, time, impact_type)
% Extracts Time of Arrival (ToA) based on impact head type
%
% Inputs:
%   signal - Sensor signal (time-domain)
%   time - Time vector corresponding to signal
%   impact_type - Type of impact head ('steel', 'silicone', 'rubber', 'soft_rubber')
%
% Output:
%   ToA - Estimated Time of Arrival

    % Select ToA extraction method based on impact head type
    switch lower(impact_type)
        case 'stl'
            ToA = extract_toa_aic(signal, time); 
        case 'sil'
            ToA = extract_toa_aic(signal, time);  
        case 'rub'
            ToA = extract_toa_aic(signal, time);
        case 'srub'
            ToA = extract_toa_aic(signal, time);
        otherwise
            error('Unknown impact type: %s', impact_type);
    end

end


function ToA = extract_toa_nset(signal, time)
% Applies Normalized Smoothed Envelope Threshold (NSET) method
% Best for high-energy impacts (steel)

    % Ensure signal is a column vector
    if isrow(signal)
        signal = signal'; % Convert to column vector
    end

    % Compute envelope of signal with zero-padding
    % padded_signal = [zeros(100,1); signal; zeros(100,1)]; % Ensure column-wise padding
    % envelope = abs(hilbert(padded_signal));
    % envelope = envelope(101:end-100); % Remove extra padding
    envelope = abs(signal);

    % Normalise by max amplitude
    norm_env = envelope / max(envelope);

    % Detect first crossing of threshold (2.5%)
    threshold = 0.025;
    idx = find(norm_env > threshold, 1, 'first');

    % Return ToA
    ToA = time(idx);
end


function ToA = extract_toa_aic(signal, time)
% Optimized Akaike Information Criterion (AIC) ToA Extraction (Fully Vectorized)

    N = length(signal);
    energy = signal.^2;  % Squared signal (energy envelope)

    % Compute cumulative sum of energy
    cumsum_energy = cumsum(energy);
    cumsum_energy_sq = cumsum(energy.^2);

    % Compute number of points k (avoiding first and last point)
    k = (2:N-1)';  

    % Compute variance using cumulative sum trick (FAST!)
    mean1 = cumsum_energy(k) ./ k;
    mean2 = (cumsum_energy(end) - cumsum_energy(k)) ./ (N - k - 1);
    
    var1 = (cumsum_energy_sq(k) ./ k) - mean1.^2; % Variance of first segment
    var2 = ((cumsum_energy_sq(end) - cumsum_energy_sq(k)) ./ (N - k - 1)) - mean2.^2; % Variance of second segment

    % Prevent log(0) issues
    var1(var1 <= 0) = 1e-10;
    var2(var2 <= 0) = 1e-10;

    % Compute AIC function
    AIC = k .* log(var1) + (N - k - 1) .* log(var2);

    % Find index of minimum AIC
    [~, idx_min] = min(AIC);
    ToA = time(idx_min);

end





%%
% function ToA = extract_toa_aic(signal, time)
% % Akaike Information Criterion (AIC) ToA Extraction
% % Best for medium-energy impacts (silicone)
% 
%     % Compute energy envelope
%     energy = signal.^2;
% 
%     % Compute AIC function
%     N = length(energy);
%     AIC = zeros(N,1);
%     for k = 2:N-1
%         AIC(k) = k * log(var(energy(1:k))) + (N-k-1) * log(var(energy(k+1:N)));
%     end
% 
%     % Find minimum of AIC function
%     [~, idx] = min(AIC);
% 
%     % Return ToA
%     ToA = time(idx);
% 
% end


% 
% function ToA = extract_toa_slta_aic(signal, time)
%     % Vectorized Short-Term / Long-Term AIC Time of Arrival (ToA) Extraction
%     %
%     % INPUT:
%     % signal   - 1D array of the original impact signal
%     % time     - Corresponding time values
%     % short_win - Window size for Short-Term Average (e.g., 50)
%     % long_win  - Window size for Long-Term Average (e.g., 500)
%     %
%     % OUTPUT:
%     % ToA      - Estimated Time of Arrival
% 
%     N = length(signal);
% 
%     % === Step 1: Compute Energy Envelope using Hilbert Transform ===
%     ST_window = 50; % Short-term window size
%     LT_window = 500; % Long-term window size
%     % === Step 2: Compute Short-Term / Long-Term Average Ratio ===
%     short_avg = movmean(abs(signal), ST_window);  % Fast response to impact
%     long_avg  = movmean(abs(signal), LT_window);   % Smooth background variations
% 
%     % Prevent division by zero
%     long_avg(long_avg <= 0) = 1e-10;
% 
%     % Compute SLTA ratio
%     SLTA_ratio = short_avg ./ long_avg;
% 
%     % === Step 3: Compute AIC on the SLTA-enhanced signal ===
% 
%     % Compute cumulative sum and squared sum
%     cumsum_signal = cumsum(SLTA_ratio);
%     cumsum_sq = cumsum(SLTA_ratio.^2);
% 
%     % Compute variance in a vectorized way
%     k = (2:N-1)';  % Avoid first and last points to prevent zero division
% 
%     var1 = (cumsum_sq(k) ./ k) - (cumsum_signal(k) ./ k).^2;
%     var2 = ((cumsum_sq(end) - cumsum_sq(k)) ./ (N - k - 1)) - ((cumsum_signal(end) - cumsum_signal(k)) ./ (N - k - 1)).^2;
% 
%     % Prevent log(0) issues
%     var1(var1 <= 0) = 1e-10;
%     var2(var2 <= 0) = 1e-10;
% 
%     % Compute AIC function
%     AIC = k .* log(var1) + (N - k - 1) .* log(var2);
% 
%     % Find index of minimum AIC
%     [~, idx_min] = min(AIC);
% 
%     % Return ToA corresponding to the minimum AIC index
%     ToA = time(idx_min);
% end
