clc
clear all 
close all
% ---------------------------- p(x) ---------------------------- %
x = linspace(0,6,1000);

plot(x, px(x),'LineWidth', 2.5);
xlabel('x [m]');
ylabel('p(x)');
grid on;

% ---------------------------- p(z|x) ---------------------------- %
z = linspace(0,6,1000);

x1 = 1.5;
x2 = 2.0;

figure;
plot(z, pz_x(z, x1), 'b', 'LineWidth', 2.5); hold on;
plot(z, pz_x(z, x2), 'r--', 'LineWidth', 2.5);
xlabel('z [m]');
ylabel('p(z | x)');
legend('x = 1.5 m', 'x = 2.0 m');
grid on;


% ---------------------------- p(z) ---------------------------- %

figure;
z = linspace(0,6,1000);
plot(z, pz(z),'LineWidth', 2.5);
xlabel('z [m]');
ylabel('p(z)');
% ---------------------------- p(x|z) ---------------------------- %

% Calculate and plot p(x|z) for z=3.1 and z=4.0
x = linspace(0, 6, 1000);  % Wide range covering posteriors

figure;
hold on;

% z = 3.1 (slightly right of prior mean=2)
pz_31 = px_z(x, 3.1);
plot(x, pz_31, 'b-', 'LineWidth', 2, 'DisplayName', 'p(x|z=3.1)');

% z = 4.0 (further right)
pz_40 = px_z(x, 4.0);
plot(x, pz_40, 'r--', 'LineWidth', 2, 'DisplayName', 'p(x|z=4.0)');

hold off;
xlabel('z'); ylabel('p(x|z)');
title('Posterior Distributions');
legend('Location', 'best');
grid on;
