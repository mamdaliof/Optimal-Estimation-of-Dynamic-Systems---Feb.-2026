%% Bayes Risk Calculator - Question 4 Solution
clc; clear; close all;

z_values = [3.1, 4.0];
x_fine = linspace(0, 6, 5000);
A = 0.05;  

figure('Position', [100 100 1400 600]);

for i = 1:length(z_values)
    z = z_values(i);
    
    % Get all estimators using unified function
    est_map  = get_estimator(x_fine, z, 'MAP');
    est_mmse = get_estimator(x_fine, z, 'MMSE');
    est_mmae = get_estimator(x_fine, z, 'MMAE');
    
    % Calculate risks for each cost function
    risk_map_mse  = quadratic_cost_risk(est_map,  z);
    risk_mmse_mse = quadratic_cost_risk(est_mmse, z);
    risk_mmae_mse = quadratic_cost_risk(est_mmae, z);
    
    risk_map_abs  = absolute_cost_risk(est_map,  z);
    risk_mmse_abs = absolute_cost_risk(est_mmse, z);
    risk_mmae_abs = absolute_cost_risk(est_mmae, z);
    
    risk_map_uni  = uniform_cost_risk(est_map,  z, A);
    risk_mmse_uni = uniform_cost_risk(est_mmse, z, A);
    risk_mmae_uni = uniform_cost_risk(est_mmae, z, A);
    
    % Original plot (kept)
    subplot(2, 3, i);
    x_plot = linspace(0, 6, 1000);
    plot(x_plot, px_z(x_plot, z), 'b-', 'LineWidth', 2);
    hold on;
    max_pdf = 1.2 * max(px_z(x_plot, z));
    plot(est_mmse, 0, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', sprintf('MMSE=%.3f', est_mmse));
    plot(est_mmae, 0, 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'DisplayName', sprintf('MMAE=%.3f', est_mmae));
    plot(est_map,  0, 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', sprintf('MAP=%.3f', est_map));
    hold off;
    title(sprintf('p(x|z=%.1f)', z)); legend('Location', 'best'); grid on;
    
    % Print results
    fprintf('=== z = %.1f ===\n', z);
    fprintf('Estimators: MAP=%.3f, MMSE=%.3f, MMAE=%.3f\n\n', est_map, est_mmse, est_mmae);
    fprintf('Quadratic Cost Risks:\n');
    fprintf('  MAP:  %.4f, MMSE: %.4f, MMAE: %.4f\n', risk_map_mse, risk_mmse_mse, risk_mmae_mse);
    fprintf('Absolute Cost Risks:\n');
    fprintf('  MAP:  %.4f, MMSE: %.4f, MMAE: %.4f\n', risk_map_abs, risk_mmse_abs, risk_mmae_abs);
    fprintf('Uniform Cost (A=%.2f) Risks:\n', A);
    fprintf('  MAP:  %.4f, MMSE: %.4f, MMAE: %.4f\n\n', risk_map_uni, risk_mmse_uni, risk_mmae_uni);
end
sgtitle('Bayes Estimators + Risk Analysis');

%% Unified Estimator Function
function x_hat = get_estimator(x_grid, z, criterion)
    pxz = px_z(x_grid, z);
    switch lower(criterion)
        case 'map'
            [~, idx] = max(pxz);
            x_hat = x_grid(idx);
        case 'mmse'
            x_hat = trapz(x_grid, x_grid .* pxz);
        case 'mmae'
            pxz_cdf = cumtrapz(x_grid, pxz);
            pxz_cdf = pxz_cdf / pxz_cdf(end);
            [~, idx] = min(abs(pxz_cdf - 0.5));
            x_hat = x_grid(idx);
        otherwise
            error('Unknown criterion: %s', criterion);
    end
end

%% Cost Function Risks (∫ C(|x̂-x|) p(x|z) dx)
function risk = quadratic_cost_risk(x_hat, z)
    integrand = @(x) (x - x_hat).^2 .* px_z(x, z);
    risk = integral(integrand, 0, 6);
end

function risk = absolute_cost_risk(x_hat, z)
    integrand = @(x) abs(x - x_hat) .* px_z(x, z);
    risk = integral(integrand, 0, 6);
end

function risk = uniform_cost_risk(x_hat, z, A)
    integrand = @(x) (1*(abs(x - x_hat) <= A)) .* px_z(x, z);
    risk = integral(integrand, 0, 6);
end
