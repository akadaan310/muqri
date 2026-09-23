## [q, xi, sg, u, nu] = fr_pot_threshold (scores, z, uq) — Peaks-Over-Threshold cut at exceedance
## risk z (Siffer et al., SPOT, KDD 2017). Mirror of Frontier.jl pot_threshold. uq defaults to 0.95.
function [q, xi, sg, u, nu] = fr_pot_threshold (scores, z, uq)
  if (nargin < 3) uq = 0.95; endif
  s = sort (scores(isfinite (scores)))(:); n = numel (s);
  if (n < 10)
    xi = 0; sg = NaN; u = NaN; nu = 0;
    if (n == 0) q = NaN; else q = s(end); endif
    return;
  endif
  u = fr_quantile7 (s, uq);
  exc = s(s > u) - u; nu = numel (exc);
  if (nu < 5) q = s(end); xi = 0; sg = NaN; return; endif
  [xi, sg] = fr_gpd_fit (exc);
  r = z * n / nu;
  if (abs (xi) < 1e-8) q = u - sg * log (r); else q = u + (sg / xi) * (r^(-xi) - 1); endif
  if (! isfinite (q)) q = s(end); endif
endfunction
