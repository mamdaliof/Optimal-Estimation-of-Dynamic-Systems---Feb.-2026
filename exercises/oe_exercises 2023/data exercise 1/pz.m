function y = pz(z)
    integrand = @(x) pz_x(z, x) .* px(x);
    y = integral(integrand, 0, 6, 'ArrayValued', true);
end
