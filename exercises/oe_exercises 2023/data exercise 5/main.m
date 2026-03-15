clc; clear; close all
%% Q1

I2 = eye(2);
Z2 = zeros(2);

F1  = 0.97 * I2;
Cw1 = 0.0016 * I2;

F = [I2, I2, Z2;
     Z2, I2, I2;
     Z2, Z2, F1];

Cw = [Z2, Z2, Z2;
      Z2, Z2, Z2;
      Z2, Z2, Cw1];

x_hat_init = zeros(6, 1);

C_init = diag([100^2, 100^2, 4^2, 4^2, 0.2^2, 0.2^2]);
format shortG
disp('F ='); disp(F)
disp('Cw ='); disp(Cw)
disp('x_hat(0|-1) ='); disp(x_hat_init)
disp('C(0|-1) ='); disp(C_init)
format short

%% Q2

H = [I2, Z2, Z2];  
Cn = 49 * I2;      
load('zradar.mat');  
N = 100;
 
x_pred_log = cell(1, N);   % x_hat(i|i-1)
x_est_log  = cell(1, N);   % x_hat(i|i)
C_pred_log = cell(1, N);   % C(i|i-1)
C_est_log  = cell(1, N);   % C(i|i)

x_pred = x_hat_init;   % x_hat(0|-1)
C_pred = C_init;       % C(0|-1)

for i = 1:N
    % Update
    % Kalman Parameters
    S = H * C_pred * H' + Cn;
    K = C_pred * H' / S;

    % Update x
    z_tilde = z(:, i) - H * x_pred;
    x_est = x_pred + K * z_tilde;

    % Updated C
    C_est = C_pred - K * S * K';

    x_est_log{i}  = x_est;
    C_est_log{i}  = C_est;
    x_pred_log{i} = x_pred;
    C_pred_log{i} = C_pred;

    % Prediction
    x_pred = F * x_est;
    C_pred = F * C_est * F' + Cw;
end

%% Q2
I2 = eye(2);
Z2 = zeros(2);
H = [I2, Z2, Z2];
Cn = 49 * I2;

load('zradar.mat');
N = 100;

x_pred_log = cell(1, N);
x_est_log  = cell(1, N);
C_pred_log = cell(1, N);
C_est_log  = cell(1, N);

x_pred = x_hat_init;
C_pred = C_init;

for i = 1:N
    % UPDATE
    S      = H * C_pred * H' + Cn;
    K      = C_pred * H' / S;
    z_tilde = z(:, i) - H * x_pred;
    x_est  = x_pred + K * z_tilde;
    C_est  = C_pred - K * S * K';

    x_pred_log{i} = x_pred;
    C_pred_log{i} = C_pred;
    x_est_log{i}  = x_est;
    C_est_log{i}  = C_est;

    % PREDICTION
    x_pred = F * x_est;
    C_pred = F * C_est * F' + Cw;
end

%% Q3 - Plot measurements, estimates, predictions
% Extract position components
z_pos     = z;                                         % 2x100 measurements
x_est_pos = cell2mat(cellfun(@(x) x(1:2), x_est_log,  'UniformOutput', false)); % 2x100
x_prd_pos = cell2mat(cellfun(@(x) x(1:2), x_pred_log, 'UniformOutput', false)); % 2x100

fig_q3 = figure('Name', 'Q3: Radar Tracking');
plot(z_pos(1,:),     z_pos(2,:),     'b-*', 'MarkerSize',4 , 'DisplayName', 'Measurements');
hold on;
plot(x_est_pos(1,:), x_est_pos(2,:), 'g-o','MarkerSize', 4,  'DisplayName', 'Estimated positions');
plot(x_prd_pos(1,:), x_prd_pos(2,:), 'r-s','MarkerSize', 4,  'DisplayName', 'Predicted positions');
axis equal;
legend('Location','best');
xlabel('\xi_1'); ylabel('\xi_2');
title('Measurements, Estimates, and Predictions');
grid on;

%% Q4 - Add uncertainty ellipses to estimated positions
fig_q4 = figure('Name', 'Q4: Uncertainty Ellipses');

ax_q3 = get(fig_q3, 'CurrentAxes');
copyobj(ax_q3, fig_q4);
ax_q4 = get(fig_q4, 'CurrentAxes');
hold(ax_q4, 'on');

theta = linspace(0, 2*pi, 100);
circle = [cos(theta); sin(theta)];

for i = 1:3:100
    C_pos = C_est_log{i}(1:2, 1:2);   
    [V, D] = eig(C_pos);              
    ellipse = V * sqrt(D) * circle;   

    center = x_est_log{i}(1:2);
    h = plot(ax_q4, center(1) + ellipse(1,:), ...
                    center(2) + ellipse(2,:), ...
                    'k-', 'LineWidth', 1);
end

title(ax_q4, 'Estimated Positions with Uncertainty Ellipses (every 3 steps)');
legend("Measurements", "Estimated positions", "Predicted positions", "Unvertainty region")

%% Q5
[K_ss, C_pred_ss, C_est_ss] = dlqe(F, eye(6), H, Cw, Cn);

disp('=== Steady State Results from dlqe ===')
disp('K_ss (steady state Kalman gain):');    disp(K_ss)
disp('C_pred_ss (steady state C(i|i-1)):'); disp(C_pred_ss)
disp('C_est_ss  (steady state C(i|i)):');   disp(C_est_ss)

disp('=== Comparison: Final iteration vs steady state ===')
disp('C_pred_log{100} (last predicted cov):'); disp(C_pred_log{100})
disp('C_est_log{100}  (last estimated cov):'); disp(C_est_log{100})
K_last = C_pred_log{100} * H' / (H * C_pred_log{100} * H' + Cn);
disp('K at i=100 (time-variant):'); disp(K_last)

% Steady-state Kalman filter 
x_pred_ss_log = cell(1, N);
x_est_ss_log  = cell(1, N);
C_pred_ss_log = cell(1, N);
C_est_ss_log  = cell(1, N);

x_pred_k = x_hat_init;

for i = 1:N
    % UPDATE 
    z_tilde  = z(:, i) - H * x_pred_k;
    x_est_k  = x_pred_k + K_ss * z_tilde;

    x_pred_ss_log{i} = x_pred_k;
    x_est_ss_log{i}  = x_est_k;
    C_pred_ss_log{i} = C_pred_ss;
    C_est_ss_log{i}  = C_est_ss;

    % PREDICTION 
    x_pred_k = F * x_est_k;
end

%% Plot 
x_est_ss_pos = cell2mat(cellfun(@(x) x(1:2), x_est_ss_log, 'UniformOutput', false));

fig_q5 = figure('Name', 'Q5: Time-variant vs Steady-state Kalman');
plot(z_pos(1,:),        z_pos(2,:),        'b.',  'MarkerSize', 10, 'DisplayName', 'Measurements');
hold on;
plot(x_est_pos(1,:),    x_est_pos(2,:),    'g-o', 'MarkerSize', 4,  'DisplayName', 'Time-variant estimate');
plot(x_est_ss_pos(1,:), x_est_ss_pos(2,:), 'm-^', 'MarkerSize', 4,  'DisplayName', 'Steady-state estimate');
axis equal; legend('Location', 'best');
xlabel('\xi_1'); ylabel('\xi_2');
title('Time-variant vs Steady-state Kalman Filter');
grid on;

%% Plot 
fig_q5e = figure('Name', 'Q5: Steady-state Uncertainty Ellipses');
ax_q5 = copyobj(get(fig_q5, 'CurrentAxes'), fig_q5e);
hold(ax_q5, 'on');

% Steady-state ellipse 
C_pos_ss = C_est_ss(1:2, 1:2);
[V_ss, D_ss] = eig(C_pos_ss);
ellipse_ss = V_ss * sqrt(D_ss) * circle;

for i = 1:3:100
    center_ss = x_est_ss_log{i}(1:2);
    h1 = plot(ax_q5, center_ss(1) + ellipse_ss(1,:), ...
                     center_ss(2) + ellipse_ss(2,:), ...
                     'k-', 'LineWidth', 1);
    h1.Annotation.LegendInformation.IconDisplayStyle = 'off';

    C_pos_tv = C_est_log{i}(1:2, 1:2);
    [V_tv, D_tv] = eig(C_pos_tv);
    ellipse_tv = V_tv * sqrt(D_tv) * circle;

    center_tv = x_est_log{i}(1:2);
    h2 = plot(ax_q5, center_tv(1) + ellipse_tv(1,:), ...
                     center_tv(2) + ellipse_tv(2,:), ...
                     'r-', 'LineWidth', 1);
    h2.Annotation.LegendInformation.IconDisplayStyle = 'off';
end

plot(ax_q5, NaN, NaN, 'k-', 'LineWidth', 1, 'DisplayName', 'Steady-state ellipses');
plot(ax_q5, NaN, NaN, 'r-', 'LineWidth', 1, 'DisplayName', 'Time-variant ellipses');
legend(ax_q5, 'Location', 'best');
title(ax_q5, 'Steady-state vs Time-variant Uncertainty Ellipses');