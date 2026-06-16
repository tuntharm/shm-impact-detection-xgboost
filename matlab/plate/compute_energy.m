function feature_value = compute_energy(filtered_signal, impact_type)
    dt = 1/2e6; % Example sampling frequency: 2 MHz

    switch impact_type
        case {'stl', 'sil'}  % Hard impacts
            feature_value = sum(filtered_signal .^ 2) * dt;

        case {'rub', 'srub'}  % Soft impacts
            envelope = abs(hilbert(filtered_signal)); % Hilbert envelope
            feature_value = sum(envelope .^ 2) * dt;

        otherwise
            warning('Unknown impact type, using default squared sum.');
            feature_value = sum(filtered_signal .^ 2) * dt;
    end
end
