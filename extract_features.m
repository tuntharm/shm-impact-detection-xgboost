function features = extract_features(time, sensor_data, impact_type, feature_types,force_N)
    num_sensors = size(sensor_data, 2);
    num_features_per_sensor = length(feature_types); % Dynamic based on input
    features = NaN(num_sensors, num_features_per_sensor); % Pre-allocate (each row = 1 sensor)

    toa_values = NaN(num_sensors, 1); % Store raw ToA values

    % Loop through each sensor
    for i = 1:num_sensors
        signal = sensor_data(:, i);
        filtered_signal = lowpass_filter(signal, time);

        % Extract features for the current sensor
        for f = 1:num_features_per_sensor
            feature_name = feature_types{f};

            % Compute the feature based on the feature name
            switch lower(feature_name)
                case 'toa'
                    toa_values(i) = compute_ToA(filtered_signal, time, impact_type); % Store ToA for later
                    features(i, f) = toa_values(i); % Temporarily store
                case 'amplitude'
                    features(i, f) = compute_amplitude(filtered_signal)/force_N;
                case 'signalenergy'
                    features(i, f) = compute_energy(filtered_signal, impact_type)/force_N;
                case 'wavelength'
                    features(i, f) = compute_wavelength(filtered_signal, time);
                case 'moderatios'
                    features(i, f) = compute_mode_ratios(filtered_signal);
                otherwise
                    warning(['Unknown feature type: ', feature_name]);
                    features(i, f) = NaN;
            end
        end
    end

    % Normalize ToA column directly in the feature matrix
    toa_idx = find(strcmpi(feature_types, 'toa')); % Find ToA column index
    if ~isempty(toa_idx) % If ToA exists
        min_ToA = min(features(:, toa_idx), [], 'omitnan'); % Get minimum for ToA column
        features(:, toa_idx) = features(:, toa_idx) - min_ToA; % Normalize directly in the matrix
    end
end

