function ModeRatios = compute_mode_ratios(signal)
    try
        [cfs, ~] = cwt(signal, 'amor'); % Wavelet Transform
        energy_per_scale = sum(abs(cfs).^2, 2); % Energy per mode
        total_energy = sum(energy_per_scale);

        if total_energy == 0
            warning('Zero total energy, returning NaN.');
            ModeRatios = NaN;
        else
            ModeRatios = max(energy_per_scale) / total_energy; % Ratio of dominant mode
        end
    catch
        warning('Error in compute_mode_ratios, returning NaN.');
        ModeRatios = NaN;
    end
end
