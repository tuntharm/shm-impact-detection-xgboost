clear, clc

%%% ==========================
%%%   MAIN INPUT
%%% ==========================

%feature_types = {'ToA', 'Amplitude', 'SignalEnergy', 'Wavelength', 'ModeRatios'};
feature_types = {'ToA', 'Amplitude', 'SignalEnergy'};
%variable_names = {'Loc', 'Loc_X', 'Loc_Y','Loc_Z'};
variable_names = {'Loc', 'Loc_X', 'Loc_Y'};


%%% ==========================


% Configure with environment variables for portable reruns.
main_folder = string(getenv("FYP_PLATE_RAW_DIR"));
if strlength(main_folder) == 0
    main_folder = "data/raw/plate";
end

output_folder = string(getenv("FYP_PLATE_PROCESSED_DIR"));
if strlength(output_folder) == 0
    output_folder = "data/processed/plate";
end

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%%% ==========================
%%%    MODE 1: SINGLE CASE
%%% ==========================
 case_mode = "single"; 
 case_name = 'stlham_p10_flat'; 
 case_folders = struct('name', case_name); 

%%% ==========================
%%%    MODE 2: AUTOMATIC MODE
%%% ==========================
% case_folders = dir(main_folder);
% case_folders = case_folders([case_folders.isdir]);
% case_folders = case_folders(~ismember({case_folders.name}, {'.', '..','Old'}));
% case_mode = "all";

%%% ==========================
%%%   PROCESS SELECTED CASES
%%% ==========================
for c = 1:length(case_folders)
    case_name = case_folders(c).name;
    case_path = fullfile(main_folder, case_name);
    file_list = dir(fullfile(case_path, '*.txt'));
    % Skip force files (containing '_f' in the filename)
    %file_list = file_list(~contains({file_list.name}, '_f'));

    %% Extracting-------------------------------------
    % Extract impact_type from case name
    impact_type_match = regexp(case_name, '(stl|sil|rub|srub)', 'match');
    if isempty(impact_type_match)
        warning(['Skipping unknown impact type in case name: ', case_name]);
        continue; % Skip this folder
    end
    impact_type = impact_type_match{1};
    
    % Extract Probe from case name
    probe_match = regexp(case_name, 'p(\d+)', 'tokens'); % Find "p" followed by a number
    
    if ~isempty(probe_match)
        probe = str2double(probe_match{1}{1}); % Convert to number
    else
        error('Probe number not found in case name.');
    end


    % Extract numeric location number and sort files
    loc_numbers = zeros(length(file_list), 1);
    for i = 1:length(file_list)
        loc_str = regexp(file_list(i).name, 'l(\d+)', 'tokens');
        if ~isempty(loc_str)
            loc_numbers(i) = str2double(loc_str{1});
        else
            loc_numbers(i) = Inf; % If no match, put at the end
        end
    end

    %% Sort files based on extracted numeric location---------
    [~, sorted_indices] = sort(loc_numbers);
    file_list = file_list(sorted_indices);

    % Initialize storage
    all_features = [];
    distance_features = [];
    force_values = [];

    %% Process each impact file
    for i = 1:length(file_list)
        filename = fullfile(case_path, file_list(i).name);
        [time, sensor_data, force_data] = read_labview_file_plate(filename);

        % Extract features
        distance = extract_distance_features_plate(filename);
        features = extract_features_plate(time, sensor_data/probe, impact_type, feature_types); % scale by probe
        force_N = max(convert_voltage_to_force(force_data,impact_type));
    
        % Debug: Print size of features----------------------------
        % disp(['File: ', file_list(i).name, ' | Features size: ', num2str(size(features, 1)), ' x ', num2str(size(features, 2))]);
        % 
        % % Check if features has the same number of columns as all_features
        % if ~isempty(all_features) && size(features, 2) ~= size(all_features, 2)
        %     error(['Dimension mismatch! File: ', file_list(i).name, ...
        %            ' | Expected columns: ', num2str(size(all_features, 2)), ...
        %            ' | Found: ', num2str(size(features, 2))]);
        % end

        %%--------------------------------------------
        % Store extracted data
        distance_features = [distance_features; distance];
        all_features = [all_features; features];
        force_values = [force_values; force_N];

        [~, filename, ~] = fileparts(filename);
        disp(['Processed file ', num2str(i), ' ', char(filename)]);

    end

    % Skip saving if no data is extracted
    if isempty(all_features)
        warning(['No data extracted for case: ', case_name]);
        continue;
    end

    % Rearrange features
    num_sensors = size(sensor_data, 2);
    num_features_per_sensor = length(feature_types); % ToA, Amplitude, Signal Energy, Wavelength, Mode Ratios
    rearranged_features = [];
    for f = 1:num_features_per_sensor
        rearranged_features = [rearranged_features, all_features(:, f:num_features_per_sensor:end)];
    end

    % Final concatenation
    final_features = [distance_features, rearranged_features, force_values];


    % Feature columns
    for f = 1:num_features_per_sensor
        for i = 1:num_sensors
            variable_names = [variable_names, {[feature_types{f} '_S' num2str(i)]}];
        end
    end

    % Append force column
    variable_names = [variable_names, {'Force_N'}];

    % Save to CSV
    output_file = fullfile(output_folder, ['Features_' case_name '.csv']);
    feature_table = array2table(final_features, 'VariableNames', variable_names);
    writetable(feature_table, output_file);

    disp(['Completed case: ', case_name, ' (Mode: ', case_mode, ')']);
    disp(['Saved to: ', output_file]);

    if case_mode == "single"
        break;
    end
    %Default Column
    variable_names = {'Loc', 'Loc_X', 'Loc_Y'};
end
