clc; clear; close all
data = load("depthgauge_data_set1.mat");

%% Q1
figure("Name","Q1")
scatter(data.x1, data.z1, 'filled')
hold on

x_max = max(data.x1)+0.5;
x_line = [0, x_max];
plot(x_line, x_line, 'r--', 'LineWidth', 1.5)

xlim([0, x_max])
ylim([0, max(data.z1)+0.1])  
xlabel('True Value (x)')
ylabel('Measurement Value (z)')
title('True vs Measured Values')
legend('Measurements', 'z=x Reference', 'Location', 'best')
grid on
hold off

%% Q2

M_xz = mean(data.x1 .* data.z1);  % Estimate of E[xz]
M_zz = mean(data.z1.^2);          % Estimate of E[z^2]


% From MMSE slide: alpha = E[xz] / E[z^2]
alpha = M_xz / M_zz;

fprintf('Linear MMSE Estimator Parameters:\n');
fprintf('alpha = %.4f\n', alpha);
fprintf('Estimator: x_hat_lMMSE(z) = %.4f * z\n\n', alpha);

figure("Name","Q2")

scatter(data.x1, data.z1, 'filled')
hold on
plot(x_line, x_line, 'r--', 'LineWidth', 1.5)

% Add linear MMSE estimator
z_line = linspace(0, x_max, 100);
x_hat_lMMSE = alpha * z_line;
plot(z_line, x_hat_lMMSE, 'b-', 'LineWidth', 2)

xlim([0, x_max])
ylim([0, max(data.z1)+0.1])  
xlabel('Measurement Value (z)')
ylabel('True Value (x) / Estimate')
title('Linear MMSE Estimator')
legend('Measurements', 'z=x Reference', 'Linear MMSE Estimator', 'Location', 'best')
grid on
hold off

%% Q3

M_x = mean(data.x1);              % Estimate of E[x]
M_z = mean(data.z1);              % Estimate of E[z]
Cov_xz = cov(data.x1, data.z1);   % Covariance matrix
Cov_xz_val = Cov_xz(1,2);         % Cov[x,z]
Var_z = var(data.z1);             % Var[z]

% Calculate alpha_opt and beta_opt using formulas from your image
alpha_opt = Cov_xz_val / Var_z;
beta_opt = M_x - alpha_opt * M_z;

fprintf('Unbiased Linear MMSE Estimator Parameters:\n');
fprintf('alpha_opt = %.4f\n', alpha_opt);
fprintf('beta_opt  = %.4f\n', beta_opt);
fprintf('Estimator: x_hat_ulMMSE(z) = %.4f * z + %.4f\n\n', alpha_opt, beta_opt);

figure("Name","Q3")
scatter(data.x1, data.z1, 'filled')
hold on
plot(x_line, x_line, 'r--', 'LineWidth', 1.5)

% Add both estimators for comparison
plot(z_line, alpha * z_line, 'b-', 'LineWidth', 2)
x_hat_ulMMSE = alpha_opt * z_line + beta_opt;
plot(z_line, x_hat_ulMMSE, 'g-', 'LineWidth', 2)

xlim([0, x_max])
ylim([0, max(data.z1)+0.1])  
xlabel('Measurement Value (z)')
ylabel('True Value (x) / Estimate')
title('Unbiased Linear MMSE Estimator')
legend('Measurements', 'z=x Reference', 'Linear MMSE', 'Unbiased Linear MMSE', 'Location', 'best')
grid on
hold off

%% Estimator Functions
function x_hat = estimator_lMMSE(z, alpha)
    x_hat = alpha * z;
end

function x_hat = estimator_ulMMSE(z, alpha_opt, beta_opt)
    x_hat = alpha_opt * z + beta_opt;
end


%% Performance Evaluation on Dataset 1 (Training Set)
fprintf('=== Performance on Dataset 1 (Training Set) ===\n\n');

% Apply linear MMSE estimator 
x_hat_lMMSE_set1 = estimator_lMMSE(data.z1, alpha);
error_lMMSE_set1 = data.x1 - x_hat_lMMSE_set1;
bias_lMMSE_set1 = mean(error_lMMSE_set1);
variance_lMMSE_set1 = var(error_lMMSE_set1);
MSE_lMMSE_set1 = mean(error_lMMSE_set1.^2);
MAE_lMMSE_set1 = mean(abs(error_lMMSE_set1));

fprintf('Linear MMSE Estimator on Dataset 1:\n');
fprintf('Bias (mean of errors):     %.6f\n', bias_lMMSE_set1);
fprintf('Variance of errors:        %.6f\n', variance_lMMSE_set1);
fprintf('MSE:                       %.6f\n', MSE_lMMSE_set1);
fprintf('MAE:                       %.6f\n\n', MAE_lMMSE_set1);

% Apply unbiased linear MMSE estimator 
x_hat_ulMMSE_set1 = estimator_ulMMSE(data.z1, alpha_opt, beta_opt);
error_ulMMSE_set1 = data.x1 - x_hat_ulMMSE_set1;
bias_ulMMSE_set1 = mean(error_ulMMSE_set1);
variance_ulMMSE_set1 = var(error_ulMMSE_set1);
MSE_ulMMSE_set1 = mean(error_ulMMSE_set1.^2);
MAE_ulMMSE_set1 = mean(abs(error_ulMMSE_set1));

fprintf('Unbiased Linear MMSE Estimator on Dataset 1:\n');
fprintf('Bias (mean of errors):     %.6f\n', bias_ulMMSE_set1);
fprintf('Variance of errors:        %.6f\n', variance_ulMMSE_set1);
fprintf('MSE:                       %.6f\n', MSE_ulMMSE_set1);
fprintf('MAE:                       %.6f\n\n', MAE_ulMMSE_set1);

fprintf('Dataset 1 Comparison:\n');
fprintf('Linear MMSE:     Bias = %.6f, Variance = %.6f, MSE = %.6f, MAE = %.6f\n', ...
    bias_lMMSE_set1, variance_lMMSE_set1, MSE_lMMSE_set1, MAE_lMMSE_set1);
fprintf('Unbiased LMMSE:  Bias = %.6f, Variance = %.6f, MSE = %.6f, MAE = %.6f\n\n', ...
    bias_ulMMSE_set1, variance_ulMMSE_set1, MSE_ulMMSE_set1, MAE_ulMMSE_set1);


%% Q4
data2 = load("depthgauge_data_set2.mat");

% Apply linear MMSE estimator
x_hat_lMMSE_set2 = estimator_lMMSE(data2.z2, alpha);
error_lMMSE = data2.x2 - x_hat_lMMSE_set2;
bias_lMMSE = mean(error_lMMSE);
variance_lMMSE = var(error_lMMSE);
MSE_lMMSE = mean(error_lMMSE.^2);
MAE_lMMSE = mean(abs(error_lMMSE));

fprintf('Q4: Linear MMSE Estimator Performance on Dataset 2:\n');
fprintf('Bias (mean of errors):     %.6f\n', bias_lMMSE);
fprintf('Variance of errors:        %.6f\n', variance_lMMSE);
fprintf('MSE:                       %.6f\n', MSE_lMMSE);
fprintf('MAE:                       %.6f\n\n', MAE_lMMSE);


%% Q5
% Apply unbiased linear MMSE estimator
x_hat_ulMMSE_set2 = estimator_ulMMSE(data2.z2, alpha_opt, beta_opt);
error_ulMMSE = data2.x2 - x_hat_ulMMSE_set2;
bias_ulMMSE = mean(error_ulMMSE);
variance_ulMMSE = var(error_ulMMSE);
MSE_ulMMSE = mean(error_ulMMSE.^2);
MAE_ulMMSE = mean(abs(error_ulMMSE));

fprintf('Q5: Unbiased Linear MMSE Estimator Performance on Dataset 2:\n');
fprintf('Bias (mean of errors):     %.6f\n', bias_ulMMSE);
fprintf('Variance of errors:        %.6f\n', variance_ulMMSE);
fprintf('MSE:                       %.6f\n', MSE_ulMMSE);
fprintf('MAE:                       %.6f\n\n', MAE_ulMMSE);

%% Comparison
fprintf('Performance Comparison:\n');
fprintf('Linear MMSE:     Bias = %.6f, Variance = %.6f, MSE = %.6f, MAE = %.6f\n', ...
    bias_lMMSE, variance_lMMSE, MSE_lMMSE, MAE_lMMSE);
fprintf('Unbiased LMMSE:  Bias = %.6f, Variance = %.6f, MSE = %.6f, MAE = %.6f\n', ...
    bias_ulMMSE, variance_ulMMSE, MSE_ulMMSE, MAE_ulMMSE);

%% Plot
figure("Name","Q4 & Q5")
s= scatter(data.x1, data.z1, 'filled')
s.MarkerFaceAlpha = 0.7;           % face transparency (0..1)
s.MarkerEdgeAlpha = 0.9;    

hold on
s= scatter(data2.x2, data2.z2, 'filled')
s.MarkerFaceAlpha = 0.7;           % face transparency (0..1)
s.MarkerEdgeAlpha = 0.9;    

plot(x_line, x_line, 'r--', 'LineWidth', 1.5)

% Add both estimators for comparison
plot(z_line, alpha * z_line, 'b-', 'LineWidth', 2)
x_hat_ulMMSE = alpha_opt * z_line + beta_opt;
plot(z_line, x_hat_ulMMSE, 'g-', 'LineWidth', 2)

xlim([0, x_max])
ylim([0, max(data.z1)+0.1])  
xlabel('Measurement Value (z)')
ylabel('True Value (x) / Estimate')
title('Unbiased Linear MMSE Estimator')
legend('train set','test set', 'z=x Reference', 'Linear MMSE', 'Unbiased Linear MMSE', 'Location', 'best')
grid on
hold off