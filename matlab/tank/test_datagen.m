clear, clc, close all
% Flat-----
%filename = "data/raw/Old/stlham_p10_flat"
% MAC------
%filename = 'data/raw/stlham_p1_tank/stlham_l1_p1_tank_1.txt'
%filename = 'data/raw/Flat/stlham_p10_flat/stlham_l1_p10_flat_1.txt'
%PC -------
filename = "data/raw/stlham_p1_tank_new/stlham_l1_p1_tank_2.txt";
%filename = "data/raw/silham_p1_tank/silham_l1_p1_tank_1.txt"
%filename = "data/raw/rubham_p10_tank/rubham_l1_p10_tank_1.txt";
%filename = "data/raw/srubham_p10_tank/srubham_l1_p10_tank_1.txt"
[time, sensor_data, force_data] = read_labview_file(filename);
variable_names = {'Loc', 'Loc_X', 'Loc_Y'};

feature_types = {'ToA', 'Amplitude', 'SignalEnergy'};

%%
 %% Extracting-------------------------------------
    % Extract impact_type from case name
    impact_type_match = regexp(filename, '(stl|sil|rub|srub)', 'match');

    impact_type = impact_type_match{1};
    
    % Extract Probe from case name
    probe_match = regexp(filename, 'p(\d+)', 'tokens'); % Find "p" followed by a number
    
    if ~isempty(probe_match)
        probe = str2double(probe_match{1}{1}); % Convert to number
    else
        error('Probe number not found in case name.');
    end


    % Extract numeric location number and sort files

    loc_str = regexp(filename, 'l(\d+)', 'tokens');
    loc_numbers = str2double(loc_str{1});
 

    % Initialize storage
    all_features = [];
    distance_features = [];
    force_values = [];

    %% Process each impact file

        [time, sensor_data, force_data] = read_labview_file(filename);

        % Extract features
        distance = extract_distance_features(filename);
        features = extract_features(time, sensor_data/probe, impact_type, feature_types); % scale by probe
        force_N = max(convert_voltage_to_force(force_data,impact_type));

