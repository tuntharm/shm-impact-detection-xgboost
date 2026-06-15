clc; clear;

% Define folder path
folder_path = "C:\Users\tunta\OneDrive - Imperial College London\Y4 work\FYP\FYP_Data\Raw_Data\plate\stlham_p1_flat";

% Get list of all `.txt` files in the folder
file_list = dir(fullfile(folder_path, '*.txt'));
file_names = {file_list.name}; % Extract filenames

% Extract numeric indices from filenames (assumes `_<number>.txt` format)
num_values = zeros(size(file_names));
for i = 1:length(file_names)
    tokens = regexp(file_names{i}, '_(\d+)\.txt$', 'tokens');
    if ~isempty(tokens)
        num_values(i) = str2double(tokens{1});
    else
        num_values(i) = Inf; % In case filename doesn't match pattern
    end
end

% Sort files numerically based on extracted numbers
[~, sorted_indices] = sort(num_values);
file_names = file_names(sorted_indices);

% Number of files per location
files_per_location = 4;
num_locations = ceil(length(file_names) / files_per_location); % Total locations

% Loop through files and rename them
for i = 1:length(file_names)
    old_name = file_names{i};  % Get the original filename
    old_path = fullfile(folder_path, old_name);

    % Compute location number (1-40)
    loc_number = ceil(i / files_per_location);
    
    % Compute within-location index (_1, _2, _3, _4)
    within_loc_idx = mod(i-1, files_per_location) + 1; 

    % Generate new filename
    new_name = sprintf('silham_l%d_p1_flat_%d.txt', loc_number, within_loc_idx);
    new_path = fullfile(folder_path, new_name);

    % Only rename if the names are different
    if ~strcmp(old_name, new_name)
        movefile(old_path, new_path);
        fprintf('Renamed: %s → %s\n', old_name, new_name);
    else
        fprintf('Skipping (already correct): %s\n', old_name);
    end
end

disp('✅ File renaming completed.');
