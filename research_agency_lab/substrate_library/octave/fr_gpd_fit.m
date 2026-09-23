## [xi, sg] = fr_gpd_fit (y) — GPD shape/scale MLE for positive excesses via Grimshaw's profile
## likelihood in theta = xi/sigma (mirror of Frontier.jl gpd_fit). Falls back to the exponential fit.
function [xi, sg] = fr_gpd_fit (y)
  v = sort (y(isfinite (y) & y > 0))(:); n = numel (v);
  if (n < 5)
    xi = 0; if (n == 0) sg = NaN; else sg = mean (v); endif; return;
  endif
  ymax = v(end); ymin = v(1);
  lo = -1 / ymax + 1e-9; hi = 2 / max (ymin, 1e-9);
  grid = [linspace(lo, -1e-9, 200), linspace(1e-9, hi, 400)];
  best = -Inf; tb = 0;
  for t = grid
    l = fr_gpd_prof (t, v, n);
    if (l > best) best = l; tb = t; endif
  endfor
  if (best == -Inf) xi = 0; sg = mean (v); return; endif
  step = (hi - lo) / 600; a = tb - step; b = tb + step;
  ph = (sqrt (5) - 1) / 2; c = b - ph * (b - a); d = a + ph * (b - a);
  for k = 1:80
    if (fr_gpd_prof (c, v, n) > fr_gpd_prof (d, v, n))
      b = d; d = c; c = b - ph * (b - a);
    else
      a = c; c = d; d = a + ph * (b - a);
    endif
  endfor
  t = (a + b) / 2;
  if (fr_gpd_prof (t, v, n) == -Inf) xi = 0; sg = mean (v); return; endif
  xi = sum (log1p (t * v)) / n; sg = xi / t;
  if (! (isfinite (xi) && isfinite (sg) && sg > 0)) xi = 0; sg = mean (v); endif
endfunction
