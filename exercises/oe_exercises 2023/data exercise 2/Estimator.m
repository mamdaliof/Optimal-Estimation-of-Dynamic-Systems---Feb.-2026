clc; clear; close all;

z_values = [3.1, 4.0];

x_fine = linspace(0, 6, 5000);

figure('Position', [100 100 1200 500]);

for i = 1:length(z_values)
    z = z_values(i);
    
    pxz = px_z(x_fine, z);
    
    % 1. MMSE Estimator: E[x|z] = ∫ x p(x|z) dx
    mmse_est = trapz(x_fine, x_fine .* pxz);
    
    % 2. MMAE Estimator: median of p(x|z)
    pxz_cdf = cumtrapz(x_fine, pxz);
    median_idx = find(pxz_cdf >= 0.5, 1)

    mmae_est = x_fine(median_idx);
    
    % 3. MAP Estimator: argmax p(x|z)
    [~, map_idx] = max(pxz);
    map_est = x_fine(map_idx);
    
    % Plot posterior + estimators
    subplot(1, 2, i);
    x_plot = linspace(0, 6, 1000);
    plot(x_plot, px_z(x_plot, z), 'b-', 'LineWidth', 2);
    hold on;
    
    % Mark estimators
    max_pdf = 1.2 * max(px_z(x_plot, z));
    plot(mmse_est, 0, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', sprintf('MMSE=%.3f', mmse_est));
    plot(mmae_est, 0, 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'DisplayName', sprintf('MMAE=%.3f', mmae_est));
    plot(map_est,  0, 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', sprintf('MAP=%.3f', map_est));
    
    hold off;
    xlabel('x'); ylabel('p(x|z)');
    title(sprintf('p(x|z=%.1f)', z));
    legend('Location', 'best'); grid on;
    
    % Display results
    fprintf('For z=%.1f:\n', z);
    fprintf('  MMSE (conditional mean) = %.3f\n', mmse_est);
    fprintf('  MMAE (median)          = %.3f\n', mmae_est);
    fprintf('  MAP (mode)             = %.3f\n', map_est);
    fprintf('\n');
end

sgtitle('Bayes Estimators: MMSE, MMAE, MAP');
