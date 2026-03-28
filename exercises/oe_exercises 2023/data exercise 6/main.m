clc;clear;close all
%% Q1
Cw_xi  = (10^-2) * eye(2);   
Cw_v   = (10^-2) * eye(2);   
Cw_a   = (10^-2) * eye(2);   
Cw_t   = 8^2;              
Cw_phi = 0.5^2;            

Cw = blkdiag(Cw_xi, Cw_v, Cw_a, Cw_t, Cw_phi);

sigma_n1 = 1;    % deg-beacon
sigma_n2 = 0.3;  % m/s-speed
sigma_n3 = 1;    % deg-compass

Cn = diag([sigma_n1^2,sigma_n2^2,sigma_n3^2]);   

disp("Cw");
disp(Cw);
disp("Cn");
disp(Cn);

x0 = [0; 0; 0; 0; 0; 0; 400; 0];
x_beacon = [5000; 10000];
C0_xi  = (10000^2) * eye(2);
C0_v   = (2^2)     * eye(2);
C0_a   = (0.04^2)  * eye(2);
C0_t   = 300^2;
C0_phi = 10^2;

C0 = blkdiag(C0_xi, C0_v, C0_a, C0_t, C0_phi);

%%  Q2 + Q5
load('z_yacht.mat');
u = [400; 45];
N = 10000;
X_est_log = zeros(8, N);
C_est_log = zeros(8, 8, N);

% NIS log 
M = length(imeas);
NIS_log = zeros(1, M);
NIS_idx = 0;

x_hat = x0;
C_hat = C0;

meas_pos = find(imeas == 1, 1);
if ~isempty(meas_pos)
    H     = Hjacobian(x_hat, x_beacon);
    S     = H * C_hat * H' + Cn;
    K     = C_hat * H' / S;
    inn   = z(:, meas_pos) - hmeas(x_hat, x_beacon);
    inn(1) = mod(inn(1) + 180, 360) - 180;
    inn(3) = mod(inn(3) + 180, 360) - 180;
    x_hat = x_hat + K * inn;
    C_hat = C_hat - K * S * K';
    % NIS at i=1
    NIS_idx = NIS_idx + 1;
    NIS_log(NIS_idx) = inn' / S * inn;
end

X_est_log(:,1)   = x_hat;
C_est_log(:,:,1) = C_hat;

for i = 1:N-1
    % Prediction
    F      = Fjacobian(x_hat);
    x_pred = fsys(x_hat, u);
    C_pred = F * C_hat * F' + Cw;

    % Update at time i+1
    meas_pos = find(imeas == i+1, 1);
    if ~isempty(meas_pos)
        H   = Hjacobian(x_pred, x_beacon);
        S   = H * C_pred * H' + Cn;
        K   = C_pred * H' / S;
        inn = z(:, meas_pos) - hmeas(x_pred, x_beacon);
        inn(1) = mod(inn(1) + 180, 360) - 180;
        inn(3) = mod(inn(3) + 180, 360) - 180;
        x_hat = x_pred + K * inn;
        C_hat = C_pred - K * S * K';
        % NIS
        NIS_idx = NIS_idx + 1;
        NIS_log(NIS_idx) = inn' / S * inn;
    else
        x_hat = x_pred;
        C_hat = C_pred;
    end

    X_est_log(:, i+1)   = x_hat;
    C_est_log(:,:, i+1) = C_hat;
end

%% Q3 
t = (0:N-1); 

% Position
figure;
plot(t, X_est_log(1,:), 'b', t, X_est_log(2,:), 'r');
xlabel('Time (s)'); ylabel('Position (m)');
title('Estimated Position vs Time');
legend('\xi_x', '\xi_y');
grid on;

% Velocity
figure;
plot(t, X_est_log(3,:), 'b', t, X_est_log(4,:), 'r');
xlabel('Time (s)'); ylabel('Velocity (m/s)');
title('Estimated Velocity vs Time');
legend('v_x', 'v_y');
grid on;

% Acceleration
figure;
plot(t, X_est_log(5,:), 'b', t, X_est_log(6,:), 'r');
xlabel('Time (s)'); ylabel('Acceleration (m/s^2)');
title('Estimated Acceleration vs Time');
legend('a_x', 'a_y');
grid on;

% Thrust and Heading
figure;
yyaxis left;
plot(t, X_est_log(7,:), 'b');
ylabel('Thrust (N)');
yyaxis right;
plot(t, X_est_log(8,:), 'r');
ylabel('Heading (deg)');
xlabel('Time (s)');
title('Estimated Thrust and Heading vs Time');
legend('Thrust t', 'Heading \phi');
grid on;


%% Q4
function draw_ellipse(mu, C2x2, color)
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    [V, D] = eig(C2x2);
    scale = V * sqrt(D);
    ellipse = scale * circle;
    plot(mu(1) + ellipse(1,:), mu(2) + ellipse(2,:), color, 'LineWidth', 1.5,  'DisplayName', 'uncertainty region');
end

ellipse_times = 1:250:N;

figure;
hold on;
plot(X_est_log(1,:), X_est_log(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated path');

meas_indices = imeas;
plot(X_est_log(1, meas_indices), X_est_log(2, meas_indices), ...
     '.m', 'MarkerSize', 15, 'DisplayName', 'Measurement update');

for k = 1:length(ellipse_times)
    idx  = ellipse_times(k);
    mu   = X_est_log(1:2, idx);
    C2x2 = C_est_log(1:2, 1:2, idx);

    if k == 1
        draw_ellipse(mu, C2x2, 'r-');
        h = findobj(gca, 'Type', 'line', 'Color', 'r');
        h(1).DisplayName = 'Uncertainty region';
    else
        draw_ellipse(mu, C2x2, 'r-');
        % suppress from legend
        hh = findobj(gca, 'Type', 'line', 'Color', 'r');
        hh(1).HandleVisibility = 'off';
    end
end

% Mark beacon position
plot(5000, 10000, 'g*', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', 'Beacon');

% % axis equal;
xlabel('x position (m)');
ylabel('y position (m)');
title('Estimated Path with Uncertainty Regions');
legend('Estimated path', 'Measurement update', 'uncertainty region', 'Beacon');
grid on;
hold off;


%% Q5

% Chi-squared 
dof       = 3;
chi5  = chi2inv(0.05, dof);   
chi95 = chi2inv(0.95, dof);  


fprintf('Chi2 boundaries (dof=3): lower=%.3f, upper=%.3f\n', chi5, chi95);

% Running mean and variance of NIS
running_mean = cumsum(NIS_log) ./ (1:M);
running_var  = zeros(1, M);
for k = 2:M
    running_var(k) = var(NIS_log(1:k));
end

% Count how many NIS values fall outside the 95% acceptance region
outside = sum(NIS_log < chi5 | NIS_log > chi95);
fprintf('NIS values outside 95%% bounds: %d / %d (%.1f%%)\n', outside, M, 100*outside/M);

% --- Plot 1: NIS values with chi-squared percentile boundaries ---
figure;
stem(1:M, NIS_log, 'b', 'filled', 'MarkerSize', 4);
hold on;
yline(chi95, 'r--', 'LineWidth', 1.5, 'DisplayName', '97.5% bound');
yline(chi5,  'g--', 'LineWidth', 1.5, 'DisplayName', '2.5% bound');
yline(dof, 'k:',  'LineWidth', 1.5, 'DisplayName', sprintf('E[NIS]=dof=%d', dof));
xlabel('Measurement index m');
ylabel('NIS');
title('Normalized Innovation Squared (NIS)');
legend('NIS', '97.5% \chi^2 bound', '2.5% \chi^2 bound', 'Expected value');
grid on;
hold off;

% --- Plot 2: Running mean of NIS ---
figure;
plot(1:M, running_mean, 'b-', 'LineWidth', 1.5);
hold on;
yline(dof, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('Expected mean = %d', dof));
xlabel('Number of measurements');
ylabel('Running mean of NIS');
title('Running Mean of NIS');
legend('Running mean', 'Expected value');
grid on;
ylim([0 dof+0.5])
hold off;

% --- Plot 3: Running variance of NIS ---
figure;
plot(2:M, running_var(2:end), 'r-', 'LineWidth', 1.5);
hold on;
yline(2*dof, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('Expected variance = %d', 2*dof));
xlabel('Number of measurements');
ylabel('Running variance of NIS');
title('Running Variance of NIS');
legend('Running variance', 'Expected value');
grid on;
ylim([0 2*dof+0.5])
hold off;
