function y = pz(z)
    integrand = @(x) pz_x(z, x) .* px(x);
    y = integral(integrand, 0, 4, 'ArrayValued', true);
end
