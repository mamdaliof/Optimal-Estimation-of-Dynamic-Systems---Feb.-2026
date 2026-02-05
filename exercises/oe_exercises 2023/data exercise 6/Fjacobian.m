function F = Fjacobian(x)
% F = Fjacobian(x) gives the 8x8 Jacobian matrix of the system equation
% evaluated at the state x.

m = 3600;
alfa_t = 1/1200;
alfa_fi = 1/200;
delta = 1;
t0 = 400;
fi0 = 45;
a = 30;
b = 5;

s = (x(3).^2 + x(4).^2).^0.5;
if s==0
    s= eps;
    x(3) = eps;
    x(4) = eps;
end

F = zeros(8);
F(1,1) = 1;
F(1,3) = delta;
F(2,2) = 1;
F(2,4) = delta; 
F(3,3) = 1;
F(3,5) = delta;
F(4,4) = 1;
F(4,6) = delta;
F(5,3) = (1/m)*(-(a*x(3)/s + 3*b*s*x(3))*x(3)-a*s-b*s^3);
F(5,4) = (-1/m)*(a*x(4)/s + 3*b*s*x(4))*x(3);
F(5,7) = (1/m)*cos(x(8)*pi/180);
F(5,8) = (-1/m)*x(7)*sin(x(8)*pi/180)*pi/180;
F(6,3) = (-1/m)*(a*x(3)/s + 3*b*s*x(3))*x(4);
F(6,4) = (1/m)*(-(a*x(4)/s + 3*b*s*x(4))*x(4)-a*s-b*s^3);
F(6,7) = (1/m)*sin(x(8)*pi/180);
F(6,8) = (1/m)*x(7)*cos(x(8)*pi/180)*pi/180;
F(7,7) = 1 - delta*alfa_t;
F(8,8) = 1 - delta*alfa_fi;


