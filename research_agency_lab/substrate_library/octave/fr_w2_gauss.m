## [loc, shape, total] = fr_w2_gauss (m1, C1, m2, C2) — closed-form W2^2 between Gaussians.
function [loc, shape, total] = fr_w2_gauss (m1, C1, m2, C2)
  loc = sum ((m1(:) - m2(:)) .^ 2);
  C2h = fr_psd_fun (C2, @sqrt);
  shape = max (trace (C1) + trace (C2) - 2 * trace (fr_psd_fun (C2h * C1 * C2h, @sqrt)), 0);
  total = loc + shape;
endfunction
