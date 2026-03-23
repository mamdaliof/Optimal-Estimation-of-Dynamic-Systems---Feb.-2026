clc; clear; close all
data = load("log.mat");
%% Q1
loc = data.xsi;
N = size(loc, 2);
v = diff(loc,1,2);
a = diff(loc, 2, 2);
x = [loc(:, 1:N-2)
     v(:,1:N-2)
     a];
% acceleration matixes and identical and zero matrixes
I2 = eye(2);
Z2 = zeros(2);
F1 = [-0.0595 -0.1530;   
      -0.0813 -0.1716];
Cw1 = 1e-3 * [ 0.1177  -0.0026;  
               -0.0026   0.0782];

% The following equation calculate next position, velocity, and
% acceleration based on the equations provided in the question
F = [I2  I2  Z2;
     Z2  I2  I2;
     Z2  Z2  F1];
Cw = blkdiag(Z2, Z2, Cw1);

%% Q2

i = 70; 
n = 100-i; 

xp = zeros(6, i+n+1);    
Cp = zeros(6, 6, i+n+1);  

xp(:, 1: i+1) = x(:, 1: i+1);   

for j = i+2 : i+1+n             
    xp(:, j)     = F * xp(:, j-1);
    Cp(:,:,j)    = F * Cp(:,:,j-1) * F' + Cw;
end

%% Q3 
fig_q3 = figure; hold on; axis equal; grid on;

plot(x(1, 1:i+n+1), x(2, 1:i+n+1), 'b-o', ...
    'MarkerSize', 3, 'LineWidth', 1.5, ...
    'DisplayName', 'Real path (0 to i=100)');

plot(xp(1, i+1:i+1+n), xp(2, i+1:i+1+n), 'r--o', ...
    'MarkerSize', 3, 'LineWidth', 1.5, ...
    'DisplayName', "Predicted path (" + i + " to " + 100 + ")");

% Mark the starting prediction point x(10)
plot(xp(1, i+1), xp(2, i+1), 'ks', ...
    'MarkerSize', 8, 'LineWidth', 2, ...
    'DisplayName', 'Start of prediction (i=10)');

xlabel('x(i)');
ylabel('y(i)');
title('Ship Path: Real vs Predicted');
legend('Location', 'best');
%% Q4
fig_q4 = figure;
ax_q3 = findobj(fig_q3, 'Type', 'axes');
ax_q4 = copyobj(ax_q3, fig_q4);
set(ax_q4, 'Position', get(ax_q3, 'Position'));
title(ax_q4, 'Ship Path: Real vs Predicted + Uncertainty Ellipses');

hold(ax_q4, 'on');
theta  = linspace(0, 2*pi, 100);
circle = [cos(theta); sin(theta)];
step   = 10;

for j = i+2 : step : i+1+n
    C_pos = Cp(1:2, 1:2, j);
    [V, D] = eig(C_pos);
    ellipse_pts = V * sqrt(D) * circle;
    center = xp(1:2, j);
    ex = center(1) + ellipse_pts(1, :);
    ey = center(2) + ellipse_pts(2, :);
    alpha = (j - (i+2)) / n;
    color = [alpha, 1-alpha, 0];
    if j == i+2
        plot(ax_q4, ex, ey, 'Color', color, 'LineWidth', 1.5, ...
            'DisplayName', 'Uncertainty ellipse');
    else
        plot(ax_q4, ex, ey, 'Color', color, 'LineWidth', 1.5, ...
            'HandleVisibility', 'off');
    end
end

j_last = i+1+n;
if mod(j_last - (i+2), step) ~= 0   % only add if not already plotted by loop
    C_pos = Cp(1:2, 1:2, j_last);
    [V, D] = eig(C_pos);
    ellipse_pts = V * sqrt(D) * circle;
    center = xp(1:2, j_last);
    ex = center(1) + ellipse_pts(1, :);
    ey = center(2) + ellipse_pts(2, :);
    plot(ax_q4, ex, ey, 'Color', [1, 0, 0], 'LineWidth', 2, ...
        'DisplayName', 'Final uncertainty ellipse (j=100)');
end

legend(ax_q4, 'Location', 'best');
