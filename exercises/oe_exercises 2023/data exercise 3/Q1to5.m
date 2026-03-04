clc; clear; close all

%% Initialization
mu_x = [10; 20];
C_x = [25 -25; -25 70]*2;
x_0 = [100; 100];
theta = 35;
sigma_dtheta =1;

%% Q1 - a 

[V, D] = eig(C_x);          
lambda = diag(D);           
a      = sqrt(lambda);      

fprintf('=== C_x Eigenvalues ===\n');
fprintf('  lambda_0 = %.4f\n', lambda(1));
fprintf('  lambda_1 = %.4f\n', lambda(2));
fprintf('=== Scaling factors ===\n');
fprintf('  a_0 = %.4f,  a_1 = %.4f\n', a(1), a(2));
fprintf('=== C_x Eigenvectors (columns of V) ===\n');
disp(V);

figure; hold on; grid on; axis equal;

draw_ellipse(mu_x, C_x, 1, 'b',"-");

plot(mu_x(1), mu_x(2), 'b+', 'MarkerSize', 12, 'LineWidth', 2); 
xlabel('x-\xi (Nm)'); ylabel('y-\eta (Nm)');
title('Prior uncertainty region (C_x)');
legend('1\sigma', 'Location', 'best');

for i = 1:2
    quiver(mu_x(1), mu_x(2), ...
           a(i)*V(1,i), a(i)*V(2,i), ...
           0, ...                        % 0 = no auto-scaling
           'r', 'LineWidth', 1.5, ...
           'MaxHeadSize', 0.3, ...
           'DisplayName', sprintf('v_%d (a=%.2f)', i, a(i)));
end
hold off
function draw_ellipse(mu, C, k, color, line_style)
    %% a
    [V, D]      = eig(C);
    a           = sqrt(diag(D));  
    t           = linspace(0, 2*pi, 500);
    unit_circle = [cos(t); sin(t)];      
    %% b
    scaled      = diag(k * a) * unit_circle;  
    %% C
    rotated     = V * scaled;               
    %% d
    ellipse_pts = rotated + mu;     
    %% e
    plot(ellipse_pts(1,:), ellipse_pts(2,:), ...
         'Color', color, 'LineStyle', line_style, 'LineWidth', 2, 'DisplayName', "Uncertainty Region");
    hold on;
end

%% Q2
t = linspace(0, 150, 500);
angles_deg  = [theta - sigma_dtheta, theta, theta + sigma_dtheta];
line_styles = {'--', '-', '--'};
line_widths = [1.2, 2, 1.2];
labels      = {'\theta - \sigma_{\Delta\theta}  (34°)', ...
               'Line of sight  \theta = 35°', ...
               '\theta + \sigma_{\Delta\theta}  (36°)'};

fig2 = figure; hold on; grid on; axis equal;
draw_ellipse(mu_x, C_x, 1, 'b', '-');
plot(x_0(1), x_0(2), 'k*', 'MarkerSize', 10, 'LineWidth', 2, ...
     'MarkerFaceColor', 'k', 'DisplayName', 'Beacon x_0');
plot(mu_x(1), mu_x(2), 'b+', 'MarkerSize', 12, 'LineWidth', 2, ...
     'DisplayName', '\mu_x');
for i = 1:3
    alpha    = deg2rad(angles_deg(i));
    xi_line  = x_0(1) + t * cos(alpha + pi);   % +pi reverses the direction
eta_line = x_0(2) + t * sin(alpha + pi);
    if i == 2
        plot(xi_line, eta_line, 'k', 'LineStyle', line_styles{i}, ...
             'LineWidth', line_widths(i), 'DisplayName', labels{i});
    else
        plot(xi_line, eta_line, 'Color', [0.5 0.5 0.5], ...
             'LineStyle', line_styles{i}, 'LineWidth', line_widths(i), ...
             'DisplayName', labels{i});
    end
end
xlabel('x-\xi (Nm)'); ylabel('y-\eta (Nm)');
title('Q2 — Line of sight and uncertainty cone');
legend('Location', 'best');
ax2 = gca;         % <-- save the axes handle for later reuse
hold off;

%% Q3 

d         = norm(x_0 - mu_x);                    
sigma_v   = d * deg2rad(sigma_dtheta);            
bar_width = 2 * sigma_v;

fprintf('=== Q3 ===\n');
fprintf('  d             = %.4f Nm\n',   d);
fprintf('  sigma_dtheta  = %.6f rad\n',  deg2rad(sigma_dtheta));
fprintf('  sigma_v       = %.4f Nm\n',   sigma_v);
fprintf('  Bar width 2*sigma_v = %.4f Nm\n', bar_width);

% direction
theta_rad = deg2rad(theta);
line_dir  = [ cos(theta_rad);  sin(theta_rad)];  
n_perp    = [ sin(theta_rad); -cos(theta_rad)];  

fig3 = figure;                                    % new empty figure
ax3  = copyobj(ax2, fig3);                        % copy all content from ax2
ax3.Position = [0.13 0.11 0.775 0.815];           % restore default axes position

title(ax3, 'Q3 — Cone + linearized bar');

axes(ax3); hold on;

for s = [-1, 1]
    start_pt = x_0 + s * sigma_v * n_perp;
    xi_bar   = start_pt(1) + t * (-line_dir(1));  
    eta_bar  = start_pt(2) + t * (-line_dir(2));
    plot(xi_bar, eta_bar, 'm-', 'LineWidth', 1, ...
         'DisplayName', sprintf('Bar edge (\\sigma_v = %.2f Nm)', sigma_v));
end

legend(ax3, 'Location', 'best');
hold off;
%% Q4 

z = -x_0(1).*sind(theta) + x_0(2).*cosd(theta);
H = [-sind(theta) cosd(theta)];
d = norm(x_0-mu_x);
zeta_v=d.*deg2rad(sigma_dtheta);
C_v = zeta_v.^2;

% Kalman
S_kalman = H*C_x*H'+C_v;
K_kalman = C_x*H'*S_kalman^-1;
x_estimated = mu_x + K_kalman * (z - H * mu_x);
C_e = C_x - K_kalman*(H*C_x);

fprintf('=== Q4 ===\n');
fprintf('  z             = %.4f Nm\n',        z);
fprintf('  H             = [%.4f  %.4f]\n',   H(1), H(2));
fprintf('  d             = %.4f Nm\n',        d);
fprintf('  sigma_v       = %.4f Nm\n',        zeta_v);
fprintf('  C_v           = %.4f Nm^2\n',      C_v);
fprintf('  S (innovation cov) = %.4f\n',      S_kalman);
fprintf('  K_kalman      = [%.4f; %.4f]\n',   K_kalman(1), K_kalman(2));
fprintf('  x_estimated   = [%.4f; %.4f] Nm\n', x_estimated(1), x_estimated(2));
fprintf('=== Posterior covariance C_e ===\n');
fprintf('  [%.4f  %.4f]\n', C_e(1,1), C_e(1,2));
fprintf('  [%.4f  %.4f]\n', C_e(2,1), C_e(2,2));

%% Q5

figure; hold on; grid on; axis equal;

draw_ellipse(mu_x, C_x, 1, 'b',"-");

plot(mu_x(1), mu_x(2), 'b+', 'MarkerSize', 12, 'LineWidth', 2); 
xlabel('x-\xi (Nm)'); ylabel('y-\eta (Nm)');
title('Prior uncertainty region (C_x)');
legend('expected location', 'Location', 'best');


draw_ellipse(x_estimated, C_e, 1, 'r',"-");

plot(x_estimated(1), x_estimated(2), 'r+', 'MarkerSize', 12, 'LineWidth', 2); 
xlabel('x-\xi (Nm)'); ylabel('y-\eta (Nm)');
title('Prior uncertainty region (C_x)');
legend('expected location', 'Location', 'best');

t = linspace(0, 150, 500);
angles_deg  = [theta - sigma_dtheta, theta, theta + sigma_dtheta];
line_styles = {'--', '-', '--'};
line_widths = [1.2, 2, 1.2];
labels      = {'\theta - \sigma_{\Delta\theta}  (34°)', ...
               'Line of sight  \theta = 35°', ...
               '\theta + \sigma_{\Delta\theta}  (36°)'};

plot(x_0(1), x_0(2), 'k^', 'MarkerSize', 10, 'LineWidth', 2, ...
     'MarkerFaceColor', 'k', 'DisplayName', 'Beacon x_0');
for i = 1:3
    alpha    = deg2rad(angles_deg(i));
    xi_line  = x_0(1) + t * cos(alpha + pi);
    eta_line = x_0(2) + t * sin(alpha + pi);
    if i == 2
    plot(xi_line, eta_line, 'k', 'LineStyle', line_styles{i}, ...
           'LineWidth', line_widths(i), 'DisplayName', labels{i});
    else
        plot(xi_line, eta_line, 'Color', [0.5 0.5 0.5], ...
             'LineStyle', line_styles{i}, 'LineWidth', line_widths(i), ...
             'DisplayName', labels{i});
    end
end
xlabel('x-\xi (Nm)'); ylabel('y-\eta (Nm)');
title('Q2 — Line of sight and uncertainty cone');
legend('Location', 'best');
ax2 = gca;         % <-- save the axes handle for later reuse
hold off;
