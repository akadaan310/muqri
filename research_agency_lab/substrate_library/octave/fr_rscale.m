## s = fr_rscale (v) — 1.4826·MAD with IQR/1.349 and SD fallbacks (mirror of Frontier.jl rscale).
function s = fr_rscale (v)
  v = v(:);
  s = 1.4826 * median (abs (v - median (v)));
  if (s > 1e-12) return; endif
  q = fr_quantile7 (v, [0.25 0.75]);
  s = (q(2) - q(1)) / 1.349;
  if (s > 1e-12) return; endif
  if (numel (v) > 1) s = std (v); else s = 0; endif
  s = max (s, 1e-9);
endfunction
