function plot_hilbert_phase(signal, time, sensor_id)

analytic_signal = hilbert(signal);
envelope = abs(analytic_signal);

% figure;
% plot(time, signal);
% hold on;
% plot(time, envelope, 'r', 'LineWidth', 1.5); % Overlay the envelope
% hold off;
% legend('Raw Signal', 'Hilbert Envelope');

instantaneous_phase = angle(analytic_signal);
instantaneous_frequency = diff(unwrap(instantaneous_phase)) / mean(diff(time));
%instantaneous_frequency = instantaneous_frequency./max(instantaneous_frequency); %Normalise

plot(time(1:end-1), instantaneous_frequency);
xlabel('Time (s)');
ylabel('Instantaneous Frequency (Hz)');
title(['Hilbert Transform - Sensor ' num2str(sensor_id)]);
grid on

end
