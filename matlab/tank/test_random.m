clear, clc
tic
%%% ==========================
%%%   MAIN INPUT
%%% ==========================

feature_types = {'ToA', 'Amplitude', 'SignalEnergy'};
variable_names = {'Loc', 'Loc_X', 'Loc_Y'};

%%% ==========================

%----WINDOW----
main_folder = "C:\Users\tc921\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data";
output_folder = "C:\Users\tc921\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Processed_Data";

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%%% ==========================
%%%    MODE 1: SINGLE CASE
%%% ==========================
case_mode = "single"; 
case_name = 'stlham_p1_tank_new'; 
case_folders = struct('name', case_name);

%%% ==========================
%%%   PROCESS SELECTED CASES
%%% ==========================
for c = 1:length(case_folders)
    case_name = case_folders(c).name;
    case_path = fullfile(main_folder, case_name);
    file_list = dir(fullfile(case_path, '*.txt'));

    %% Extracting-------------------------------------
    impact_type_match = regexp(case_name, '(stl|sil|rub|srub)', 'match');
    if isempty(impact_type_match)
        warning(['Skipping unknown impact type in case name: ', case_name]);
        continue;
    end
    impact_type = impact_type_match{1};
    
    probe_match = regexp(case_name, 'p(\d+)', 'tokens');
    if ~isempty(probe_match)
        probe = str2double(probe_match{1}{1});
    else
        error('Probe number not found in case name.');
    end

    loc_numbers = zeros(length(file_list), 1);
    for i = 1:length(file_list)
        loc_str = regexp(file_list(i).name, 'l(\d+)', 'tokens');
        if ~isempty(loc_str)
            loc_numbers(i) = str2double(loc_str{1});
        else
            loc_numbers(i) = Inf;
        end
    end

    [~, sorted_indices] = sort(loc_numbers);
    file_list = file_list(sorted_indices);

    % Preallocate storage for parallel execution
    num_files = length(file_list);
    
    % Get number of sensors and features
    [~, sensor_data, ~] = read_labview_file(fullfile(case_path, file_list(1).name));
    num_sensors = size(sensor_data, 2);
    num_features_per_sensor = length(feature_types);

    distance_features = zeros(num_files, 3); % Preallocate for (Loc, Loc_X, Loc_Y)
    force_values = zeros(num_files, 1); % Preallocate force values
    all_features = cell(num_files, 1); % Use cell array for parallel storage

    %% Use Parallel Loop (parfor)
    parfor i = 1:num_files
        filename = fullfile(case_path, file_list(i).name);
        [time, sensor_data, force_data] = read_labview_file(filename);

        % Extract features
        distance = extract_distance_features(filename); % Now 1×3
        force_N = max(convert_voltage_to_force(force_data, impact_type));
        features = extract_features(time, sensor_data / probe, impact_type, feature_types,force_N); % (num_sensors × num_features)

        % Store extracted data
        distance_features(i, :) = distance; % Matches correct shape (1×3)
        all_features{i} = features(:)'; % Store as row vector inside a cell
        force_values(i) = force_N;

        [~, filename, ~] = fileparts(filename);
        disp(['Processed file ', num2str(i), ' ', char(filename)]);
    end

    % Skip saving if no data is extracted
    if isempty(all_features)
        warning(['No data extracted for case: ', case_name]);
        continue;
    end

    % Convert cell array to numeric matrix
    all_features_matrix = cell2mat(all_features);

    % Combine final dataset
    final_features = [distance_features, all_features_matrix, force_values];

    % Adjust column names to match grouped feature structure
    feature_column_names = {};
    for f = 1:num_features_per_sensor
        for s = 1:num_sensors
            feature_column_names = [feature_column_names, {[feature_types{f} '_S' num2str(s)]}];
        end
    end

    variable_names = [{'Loc', 'Loc_X', 'Loc_Y'}, feature_column_names, {'Force_N'}];

    % Save to CSV
    output_file = fullfile(output_folder, ['Features_' case_name '.csv']);
    feature_table = array2table(final_features, 'VariableNames', variable_names);
    writetable(feature_table, output_file);

    disp(['Completed case: ', case_name, ' (Mode: ', case_mode, ')']);
    disp(['Saved to: ', output_file]);

    if case_mode == "single"
        break;
    end
    % Default Column
    variable_names = {'Loc', 'Loc_X', 'Loc_Y'};
end
toc
