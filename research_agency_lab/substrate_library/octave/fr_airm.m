## d = fr_airm (C1, C2) — affine-invariant Riemannian distance ||log(C1^-1/2 C2 C1^-1/2)||_F.
function d = fr_airm (C1, C2)
  Ci = fr_psd_fun (C1, @(x) 1 ./ sqrt (max (x, 1e-300)));
  M = Ci * C2 * Ci;
  l = eig ((M + M') / 2);
  d = sqrt (sum (log (max (l, 1e-300)) .^ 2));
endfunction
