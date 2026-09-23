## l = fr_gpd_prof (t, v, n) — Grimshaw profile log-likelihood at theta = t for excesses v.
function l = fr_gpd_prof (t, v, n)
  if (t == 0 || 1 + t * v(end) <= 1e-12) l = -Inf; return; endif
  u = 1 + t * v;
  if (any (u <= 1e-12)) l = -Inf; return; endif
  s = sum (log (u)); xi = s / n;
  if (abs (xi) < 1e-12 || xi / t <= 0) l = -Inf; return; endif
  l = -n * log (xi / t) - (1 + 1 / xi) * s;
endfunction
