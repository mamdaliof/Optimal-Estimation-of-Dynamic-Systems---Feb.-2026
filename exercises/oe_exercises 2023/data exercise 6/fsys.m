function xn = fsys(x,u,Cw)
% xnext = fsys(x,u) system function of a yacht
%
% input:
% x: 8D current state vector consisting of:
%    - postition (2D)
%    - velocity (2D)
%    - acceleration (2D)
%    - actual trust (1D)
%    - actual heading (1D)
% u: 2D current control input vector consisting of:
%    - desired trust 
%    - desired heading
% output:
% xnext: the next state based on the current state and current control.
%
% optional:
% xnext = fsys(x,u,Cw) calculates the next state with addition of white process
% noise with covariance matrix Cw. This 8x8 can be any valid covariance
% matrix, but usually the process noise is only active on the actual trust
% and heading. This can be accomplished by having zeros everywhere except
% in the last two diagonal elements.
% Note: if x is an 8xN matrix, then each column in x is regarded as a state
% vector, and the resulting next states are calculated in parallel. That
% is, xnect will also be a 8xN matrix.

mass = 3600;
delta = 1;                              % sampling period
alfa_t = 1/1200;                        % AR constant thrust
alfa_fi = 1/200;                        % AR constant course

t0 = u(1);                              % trust control input
fi0 = u(2);                             % heading control input

df = 180*atan2(sin(pi*(x(8,:)-fi0)/180),cos(pi*(x(8,:)-fi0)/180))/pi;
speed = (x(3,:).^2 + x(4,:).^2).^0.5;
heading = [cos(x(8,:)*pi/180); sin(x(8,:)*pi/180)]; 
xn(1:2,:) = x(1:2,:) + delta * x(3:4,:);
xn(3:4,:) = x(3:4,:) + delta * x(5:6,:);
xn(5:6,:) = ((ones(2,1)*x(7,:)).*heading - (ones(2,1)* d(speed)).*x(3:4,:))/mass;
xn(7,:) = x(7,:) - delta*alfa_t *(x(7,:) - t0);
xn(8,:) = x(8,:) - delta*alfa_fi*df;
xn(8,:) = 180*atan2(sin(pi*(xn(8,:))/180),cos(pi*(xn(8,:))/180))/pi;

if nargin>2
    xn = xn + Cw^0.5 * randn(size(x));
end
end



function d = d(speed)
a = 30;
b = 5;
d = a*speed + b * speed.^3;
end
