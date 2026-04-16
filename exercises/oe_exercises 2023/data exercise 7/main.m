clc; clear; close all
%% Q0
data = load('z_yacht2.mat');
I = length(data.fi0);      % Length of the sequence

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

x0 = [0; 0; 0; 0; 0; 0; 400; 0];
x_beacon = [5000; 10000];

C0_xi  = (100^2) * eye(2);
C0_v   = (2^2)     * eye(2);
C0_a   = (0.04^2)  * eye(2);
C0_t   = 300^2;
C0_phi = 10^2;
C0 = blkdiag(C0_xi, C0_v, C0_a, C0_t, C0_phi);

%% Q1 + Q2
% Listin 9.10
Ncond = 1000;
M = 8;              % Dimension of state vector
N = 3;              % Dimension of measurement vector


x_preds = x0 * ones(1,Ncond) + chol(C0)' * randn(8, Ncond);
invCn = inv(Cn);
x_est_log  = zeros(8, I);
C_est_log  = zeros(8, 8, I);
Keff_log = zeros(1, length(data.imeas));

for i = 1:I


    meas_pos = find(data.imeas == i, 1);
    if ~isempty(meas_pos)
        % Generate predicted meas. representing p(z(i)|Z(i-1))
        Zs = hmeas(x_preds, x_beacon,Cn);

        % Get uniform distributed rv
        u(1,i) = sum((Zs(1,:) < data.z(1,meas_pos)))/Ncond;
        u(2,i) = sum((Zs(2,:) < data.z(2,meas_pos)))/Ncond;
        u(3,i) = sum((Zs(3,:) < data.z(3,meas_pos)))/Ncond;


        % Update
        res = hmeas(x_preds, x_beacon) - data.z(:, meas_pos) * ones(1, Ncond); % Residuals
        res(1,:) = mod(res(1,:) + 180, 360) - 180;  % bearing
        res(3,:) = mod(res(3,:) + 180, 360) - 180;  % heading
        W = exp(-0.5*sum(res.*(invCn*res))');      % Weights
        if (sum(W)==0), error('process did not converge'); end
        W = W/sum(W); CumW = cumsum(W);
        Keff_log(meas_pos) = 1 / sum(W.^2);

        % estimate
        x_est = x_preds * W;
        x_est_log(:, i) = x_est;
        dev = x_preds - x_est * ones(1, Ncond);   % 8×Ncond deviations
        C_est_log(:,:,i) = (dev .* (ones(8,1) * W')) * dev';  % weighted outer products, 8×8

        % Resample
        % Find an index permutation using golden rule root finding
        for j = 1:Ncond
            R = rand; ja = 1; jb = Ncond;
            while (ja < jb-1)
                jx = floor(jb-0.382*(jb-ja));
                fa = R-CumW(ja); fb = R-CumW(jb); fxx = R-CumW(jx);
                if (fb*fxx < 0), ja = jx; else, jb = jx; end
            end
            ind(j) = jb;
        end

        Ys = x_preds(:, ind);
        % Predict
        x_preds = fsys(Ys,[400; data.fi0(i)],Cw);
    else
        % Predict
        x_est = mean(x_preds, 2);         % current predicted mean
        x_est_log(:, i) = x_est;          % store at i
        F = Fjacobian(x_est);
        C_est_log(:,:,i) = F * C_est_log(:,:,i-1) * F' + Cw;
        x_preds = fsys(x_preds, [400; data.fi0(i)], Cw);

    end


end


%% Q1 Plot
figure;
hold on;
plot(x_est_log(1,:), x_est_log(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated path');
plot(x_beacon(1), x_beacon(2), 'g*', 'MarkerSize', 12, 'DisplayName', 'Beacon');
plot(x_est_log(1,1), x_est_log(2,1), 'ko', 'MarkerSize', 8, 'DisplayName', 'Start');

% Mark measurement update points
meas_times = data.imeas;
plot(x_est_log(1, meas_times), x_est_log(2, meas_times), ...
    'm.', 'MarkerSize', 12, 'DisplayName', 'Measurement update');

xlabel('x position (m)');
ylabel('y position (m)');
title('Q1 – Estimated Path (Particle Filter)');
legend; grid on; hold off;

%% Q2 Plot
function draw_ellipse(mu, C2x2, color)
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    [V, D] = eig(C2x2);
    ellipse = V * sqrt(D) * circle;
    plot(mu(1) + ellipse(1,:), mu(2) + ellipse(2,:), color, 'LineWidth', 1.2);
end

ellipse_times = 1:250:I;

figure;
hold on;
plot(x_est_log(1,:), x_est_log(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Estimated path');
plot(x_beacon(1), x_beacon(2), 'g*', 'MarkerSize', 12, 'DisplayName', 'Beacon');

for k = 1:length(ellipse_times)
    idx = ellipse_times(k);
    mu   = x_est_log(1:2, idx);
    C2x2 = squeeze(C_est_log(1:2, 1:2, idx));
    if k == 1
        draw_ellipse(mu, C2x2, 'r-');
        h = findobj(gca, 'Type', 'line', 'Color', [1 0 0]);
        if ~isempty(h), h(1).DisplayName = 'Uncertainty ellipse'; end
    else
        draw_ellipse(mu, C2x2, 'r-');
        hh = findobj(gca, 'Type', 'line', 'Color', [1 0 0]);
        if ~isempty(hh), hh(1).HandleVisibility = 'off'; end
    end
end

xlabel('x position (m)');
ylabel('y position (m)');
title('Q2 – Estimated Path with Uncertainty Ellipses');
legend; grid on; hold off;

%% Q3 Plot
meas_times = data.imeas;
u_meas = u(:, meas_times);   % 3 × number_of_measurements

figure;
hold on;
plot(1:length(meas_times), u_meas(1,:), 'r.', 'MarkerSize', 8, 'DisplayName', 'u_1 (bearing)');
plot(1:length(meas_times), u_meas(2,:), 'b.', 'MarkerSize', 8, 'DisplayName', 'u_2 (speed)');
plot(1:length(meas_times), u_meas(3,:), 'g.', 'MarkerSize', 8, 'DisplayName', 'u_3 (heading)');
ylim([-0.1 1.1])
h = yline(0, 'k--', 'LineWidth', 1);
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
h =yline(1, 'k--', 'LineWidth', 1);
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
yline(0.5, 'k:', 'LineWidth', 1, 'DisplayName', 'Expected mean');
xlabel('Measurement index');
ylabel('u_n(i)');
title('Q3 – Test Variables u_n (should be uniform on [0,1])');
legend; grid on; hold off;
%% Q3 – Histograms + Autocorrelation
u_labels = {'u_1 (bearing)', 'u_2 (speed)', 'u_3 (heading)'};
colors   = {'r', 'b', 'g'};
maxLag   = 20;   % number of lags for autocorrelation

figure('Name', 'Q3 – Histograms of Test Variables');
for n = 1:3
    vals = u_meas(n, :);           % 1 × number_of_measurements
    mu_n  = mean(vals);
    var_n = var(vals);

    subplot(3, 1, n);
    histogram(vals, 10, 'Normalization', 'pdf', ...
              'FaceColor', colors{n}, 'FaceAlpha', 0.6, 'EdgeColor', 'k');
    hold on;
    % Reference uniform PDF line
    yline(1, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Uniform PDF = 1');
    xline(mu_n, 'm-', 'LineWidth', 1.5, 'DisplayName', sprintf('Mean = %.3f', mu_n));
    hold off;
    xlim([0 1]);
    xlabel('Value');
    ylabel('PDF');
    title(sprintf('%s\n\\mu = %.3f,  \\sigma^2 = %.4f', u_labels{n}, mu_n, var_n));
    legend('Location', 'best');
    grid on;
end
sgtitle('Q3 – Histograms of Test Variables u_n (Uniformity Check)');

%% Q3 – Autocorrelation of Test Variables
figure('Name', 'Q3 – Autocorrelation of Test Variables');
for n = 1:3
    vals = u_meas(n, :);
    % Normalised autocorrelation (zero-mean)
    vals_zm = vals - mean(vals);
    acf_full = xcorr(vals_zm, maxLag, 'coeff');   % symmetric, length 2*maxLag+1
    lags     = -maxLag:maxLag;

    % 95% confidence bounds for white noise: ±1.96/sqrt(N)
    N_meas = length(vals);
    conf   = 1.96 / sqrt(N_meas);

    subplot(1, 3, n);
    stem(lags, acf_full, colors{n}, 'filled', 'MarkerSize', 3); hold on;
    yline( conf, 'k--', 'LineWidth', 1.2, 'DisplayName', '95% CI');
    yline(-conf, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    yline(0, 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    hold off;
    xlabel('Lag');
    ylabel('ACF');
    title(sprintf('ACF – %s', u_labels{n}));
    legend('ACF', '95% CI', 'Location', 'best');
    grid on;
end
sgtitle('Q3 – Autocorrelation of Test Variables u_n');
%% Q4 Plot
figure;
plot(1:length(meas_times), Keff_log, 'b-o', 'MarkerSize', 4, 'LineWidth', 1.2);
hold on;
yline(Ncond, 'k--', 'DisplayName', sprintf('Max K_{eff} = %d', Ncond));
yline(Ncond/10, 'r--', 'DisplayName', sprintf('10%% threshold = %d', Ncond/10));
xlabel('Measurement index');
ylabel('K_{eff}');
title('Q4 – Effective Number of Particles');
legend; grid on; hold off;

%% Q6 
X_est_ekf  = zeros(8, I);
C_est_ekf  = zeros(8, 8, I);
NIS_ekf    = zeros(1, length(data.imeas));
NIS_idx    = 0;

x_hat = x0;
C_hat = C0;

meas_pos = find(data.imeas == 1, 1);
if ~isempty(meas_pos)
    H      = Hjacobian(x_hat, x_beacon);
    S      = H * C_hat * H' + Cn;
    K      = C_hat * H' / S;
    inn    = data.z(:, meas_pos) - hmeas(x_hat, x_beacon);
    inn(1) = mod(inn(1) + 180, 360) - 180;
    inn(3) = mod(inn(3) + 180, 360) - 180;
    x_hat  = x_hat + K * inn;
    C_hat  = C_hat - K * S * K';
    NIS_idx = NIS_idx + 1;
    NIS_ekf(NIS_idx) = inn' / S * inn;
end
X_est_ekf(:, 1)    = x_hat;
C_est_ekf(:, :, 1) = C_hat;

for i = 1:I-1
    u_i    = [400; data.fi0(i)];   % <-- time-variant phi0
    F      = Fjacobian(x_hat);
    x_pred = fsys(x_hat, u_i);
    C_pred = F * C_hat * F' + Cw;

    meas_pos = find(data.imeas == i+1, 1);
    if ~isempty(meas_pos)
        H      = Hjacobian(x_pred, x_beacon);
        S      = H * C_pred * H' + Cn;
        K      = C_pred * H' / S;
        inn    = data.z(:, meas_pos) - hmeas(x_pred, x_beacon);
        inn(1) = mod(inn(1) + 180, 360) - 180;
        inn(3) = mod(inn(3) + 180, 360) - 180;
        x_hat  = x_pred + K * inn;
        C_hat  = C_pred - K * S * K';
        NIS_idx = NIS_idx + 1;
        NIS_ekf(NIS_idx) = inn' / S * inn;
    else
        x_hat = x_pred;
        C_hat = C_pred;
    end
    X_est_ekf(:, i+1)    = x_hat;
    C_est_ekf(:, :, i+1) = C_hat;
end
NIS_ekf = NIS_ekf(1:NIS_idx);   % trim unused entries
M_ekf   = NIS_idx;

%% Q6 Plot 1 – Path + Uncertainty Ellipses
ellipse_times_ekf = 1:250:I;
figure; hold on;
plot(X_est_ekf(1,:), X_est_ekf(2,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'EKF path');
plot(X_est_ekf(1, data.imeas), X_est_ekf(2, data.imeas), ...
     '.m', 'MarkerSize', 15, 'DisplayName', 'Measurement update');
for k = 1:length(ellipse_times_ekf)
    idx  = ellipse_times_ekf(k);
    mu   = X_est_ekf(1:2, idx);
    C2x2 = squeeze(C_est_ekf(1:2, 1:2, idx));
    draw_ellipse(mu, C2x2, 'r-');
    h = findobj(gca, 'Type', 'line', 'Color', [1 0 0]);
    if k == 1
        h(1).DisplayName = 'Uncertainty region';
    else
        h(1).HandleVisibility = 'off';
    end
end
plot(x_beacon(1), x_beacon(2), 'g*', 'MarkerSize', 12, 'DisplayName', 'Beacon');
xlabel('x position (m)'); ylabel('y position (m)');
title('Q6 – EKF Path + Uncertainty Regions (time-variant \phi_0)');
legend; grid on; hold off;

%% Q6 Plot 2 – NIS
dof   = 3;
chi5  = chi2inv(0.05, dof);
chi95 = chi2inv(0.95, dof);
outside_ekf = sum(NIS_ekf < chi5 | NIS_ekf > chi95);
fprintf('Q6 EKF – NIS outside 95%% bounds: %d / %d (%.1f%%)\n', ...
        outside_ekf, M_ekf, 100*outside_ekf/M_ekf);

figure;
stem(1:M_ekf, NIS_ekf, 'b', 'filled', 'MarkerSize', 4); hold on;
yline(chi95, 'r--', 'LineWidth', 1.5, 'DisplayName', '97.5% \chi^2 bound');
yline(chi5,  'g--', 'LineWidth', 1.5, 'DisplayName', '2.5% \chi^2 bound');
yline(dof,   'k:',  'LineWidth', 1.5, 'DisplayName', sprintf('E[NIS] = %d', dof));
xlabel('Measurement index'); ylabel('NIS');
title('Q6 – NIS of EKF (time-variant \phi_0)');
legend; grid on; hold off;

% Running mean of NIS
running_mean_ekf = cumsum(NIS_ekf) ./ (1:M_ekf);
figure;
plot(1:M_ekf, running_mean_ekf, 'b-', 'LineWidth', 1.5); hold on;
yline(dof, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('Expected mean = %d', dof));
xlabel('Number of measurements'); ylabel('Running mean of NIS');
title('Q6 – Running Mean of NIS (EKF)');
legend; grid on; ylim([0 dof+1]); hold off;
















