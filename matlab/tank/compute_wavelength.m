function Wavelength = compute_wavelength(signal, time)
    try
        fs = 1 / mean(diff(time)); % Sampling frequency
        [cfs, freqs] = cwt(signal, 'amor', fs); % Continuous Wavelet Transform

        % Ensure valid frequency data
        if isempty(freqs) || isempty(cfs)
            warning('Error in compute_wavelength: Empty frequency data, returning NaN.');
            Wavelength = NaN;
            return;
        end

        % Find dominant frequency safely
        [~, row_idx] = max(max(abs(cfs), [], 2)); % Find row index of max value

        if row_idx > length(freqs) || row_idx < 1
            warning('Error in compute_wavelength: row_idx out of range, returning NaN.');
            Wavelength = NaN;
        else
            dominant_freq = freqs(row_idx);
            Wavelength = 1 / dominant_freq; % Convert to wavelength
        end
    catch
        warning('Exception in compute_wavelength, returning NaN.');
        Wavelength = NaN;
    end
end
