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
yline(0, 'k--', 'LineWidth', 1);
yline(1, 'k--', 'LineWidth', 1);
yline(0.5, 'k:', 'LineWidth', 1, 'DisplayName', 'Expected mean');
xlabel('Measurement index');
ylabel('u_n(i)');
title('Q3 – Test Variables u_n (should be uniform on [0,1])');
legend; grid on; hold off;

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