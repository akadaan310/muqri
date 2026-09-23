## fr_crosscheck (dir) — recompute the frontier layer from Julia's fixtures and report the largest
## relative discrepancy per quantity. Exits non-zero if any exceeds 1e-6.
function fr_crosscheck (dir)
  rd = @(n) dlmread (fullfile (dir, [n ".csv"]), ",");
  rel = @(a, b) norm (a(:) - b(:)) / max (norm (b(:)), 1e-12);
  X = rd ("X"); [mu, S] = fr_ogk (X);
  res.ogk_mu = rel (mu, rd ("ogk_mu")); res.ogk_S = rel (S, rd ("ogk_S"));
  M = rd ("M"); w = rd ("w"); k = numel (w);
  Cs = cell (1, k); ms = cell (1, k);
  for i = 1:k Cs{i} = rd (sprintf ("C%d", i)); ms{i} = M(:, i); endfor
  [bm, bC] = fr_bw_barycenter (ms, Cs, w);
  res.bary_m = rel (bm, rd ("bary_m")); res.bary_C = rel (bC, rd ("bary_C"));
  res.karcher = rel (fr_karcher (Cs, w), rd ("karcher"));
  s = rd ("scalars"); sc = rd ("scores");
  [l, sh, t] = fr_w2_gauss (ms{1}, Cs{1}, ms{2}, Cs{2});
  res.w2 = rel ([l sh t], s(1:3)');
  res.airm = rel (fr_airm (Cs{1}, Cs{2}), s(4));
  res.conformal = rel ([fr_conformal(sc, 0.2) fr_conformal(sc, 0.05)], s(5:6)');
  res.sinkhorn = rel (fr_sinkhorn_div (rd ("hist_a"), rd ("hist_b"), rd ("ground")), s(7));
  tail = rd ("tail"); e = rd ("evt");
  [xi, sg] = fr_gpd_fit (tail);
  res.gpd_xi = rel (xi, e(1)); res.gpd_sigma = rel (sg, e(2));
  [qa, ~, ~, ua, nua] = fr_pot_threshold (tail, 1e-2);
  qb = fr_pot_threshold (tail, 1e-3);
  res.pot_q = rel ([qa qb], e(3:4)');
  res.pot_u = rel ([ua nua], e(5:6)');
  f = fieldnames (res); worst = 0;
  for i = 1:numel (f)
    printf ("%-12s rel.err %.2e\n", f{i}, res.(f{i})); worst = max (worst, res.(f{i}));
  endfor
  printf ("WORST %.2e -> %s\n", worst, ifelse_str (worst < 1e-6));
  if (worst >= 1e-6) exit (1); endif
endfunction
function s = ifelse_str (ok)
  if (ok) s = "AGREE"; else s = "DISAGREE"; endif
endfunction
