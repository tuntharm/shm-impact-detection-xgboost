function plot_signal_processing(time, raw_signal, filtered_signal, sensor_num, ToA)
    plot(time, raw_signal, 'r', 'DisplayName', 'Raw Signal'); hold on;
    plot(time, filtered_signal, 'b', 'DisplayName', 'Filtered Signal');
%%
    % ToA = compute_ToA(filtered_signal, time, 'stl');
    % ToA_AIC = compute_ToA(filtered_signal, time, 'rub');
    % ToA_AIC_STLT = compute_ToA(filtered_signal, time, 'srub');
    xline(ToA, '--k', 'DisplayName', 'ToA');
    % xline(ToA_AIC, '--r', 'DisplayName', 'ToA AIC');
    % xline(ToA_AIC_STLT, '--g', 'DisplayName', 'ToA S/L TA AIC');

    
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    title(['Sensor ' num2str(sensor_num) ' Signal Processing']);
    legend;
    grid on;
end
