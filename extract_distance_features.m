function distance_features = extract_distance_features(filename)
    % Extract impact location number from filename (assumes "l<number>" format)
    loc_str = regexp(filename, 'l(\d+)', 'tokens');
    if ~isempty(loc_str)
        loc = str2double(loc_str{1});
    else
        warning(['Unable to extract impact location from: ', filename]);
        distance_features = NaN;
        return;
    end
    % Map impact location number to (X, Y,Z) coordinates
    [x, y] = map_location_to_xy(loc);
    distance_features = [loc, x, y];



%% === OLD 3D Cooridnate Export ===
   %  sensorxyz = [
   %  0,      11.55,     0;
   %  11.55,     0,      0;
   %  0,     -11.55,     0;
   % -11.55,     0,      0
   % 0,      11.55,     45;
   %  11.55,     0,      45;
   %  0,     -11.55,     45;
   % -11.55,     0,      45;];

    % Map impact location number to (X, Y,Z) coordinates
    % [x, y, z] = map_location_to_xyz(loc);
    % distance_features = [loc, x, y, z];


    % Debugging Output
    %disp(['Debug: Extracted Distance Features for ', filename]);
    % disp(distance_features);
end

%%% ==========================
%%%  FUNCTION: Map Location to X, Y Coordinates
%%% ==========================
% function [x, y, z] = map_location_to_xyz(loc)
%     if isnan(loc)
%         x = NaN; y = NaN; z = NaN;
%         return;
%     end
% 
%     % Define X and Y mappings based on given logic
%     if any(loc == [1, 2, 3, 4, 5,21,22,23,24,25]), x = 0;
%     elseif any(loc == [6, 7, 8, 9, 10,16, 17, 18, 19, 20]), x = -8.166;
%     elseif any(loc == [11, 12, 13, 14, 15]), x = -11.55;
%     elseif any(loc == [26,27,28,29,30,36,37,38,39,40]), x = 8.166;
%     elseif any(loc == [31, 32, 33, 34, 35]), x = 11.55;    
% 
%     else, x = NaN;
%     end
% 
%     if any(loc == [1, 2, 3, 4, 5]), y = 11.55;
%     elseif any(loc == [6, 7, 8, 9, 10,36,37,38,39,40]), y = 8.166;
%     elseif any(loc == [11, 12, 13, 14, 15,31, 32, 33, 34, 35]), y = 0;
%     elseif any(loc == [16, 17, 18, 19, 20,26,27,28,29,30]), y = -8.166;
%     elseif any(loc == [21,22,23,24,25]), y = -11.55;
%     else, y = NaN;
%     end
%     if any(loc == [1, 6,11,16,21,26,31,36]), z = 7.5;
%     elseif any(loc == [2,7,12,17,22,27,32,37]), z = 7.5*2;
%     elseif any(loc == [3,8,13,18,23,28,33,38]), z = 7.5*3;
%     elseif any(loc == [4,9,14,19,24,29,34,39]), z = 7.5*4;
%     elseif any(loc == [5,10,15,20,25,30,35,40]), z = 7.5*5;
%     else, z = NaN;
%     end
% 
% end

%%
%%% ==========================
%%%  FUNCTION: Map Location to X, Y Coordinates
%%% ==========================
function [x, y] = map_location_to_xy(loc)
    if isnan(loc)
        x = NaN; y = NaN; 
        return;
    end

    % Define X and Y mappings based on given logic
    if mod(loc,5) == 0
        x = -7.5*5;
    else 
        x = -7.5*mod(loc,5);
    end
    yloc = floor((loc-1)/5);
    y = -yloc*2*pi*11.55/8;

end
