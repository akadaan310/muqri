## fr_real_crosscheck (edir, anchor, aw) — rebuild every rule key's Bures–Wasserstein reference from the
## clouds exported by `frontier.jl --export EDIR` and compare each reciter's (loc2, airm, total2) with
## Julia's. Reference set = rows flagged is_reference (anchor + peers). Exit 1 if rel.err > 1e-6.
function fr_real_crosscheck (edir, anchor, aw)
  if (nargin < 2) anchor = "Husary_128kbps"; endif
  if (nargin < 3) aw = 0.5; endif
  d = dir (fullfile (edir, "expected__*.csv")); files = {d.name}; worst = 0;
  for f = 1:numel (files)
    key = regexprep (files{f}, '^expected__(.*)\.csv$', '$1');
    fid = fopen (fullfile (edir, files{f})); c = textscan (fid, "%s %f %f %f %f", "Delimiter", ","); fclose (fid);
    names = c{1}; isref = c{2}; J = [c{3} c{4} c{5}];
    X = cell (numel (names), 1);
    for i = 1:numel (names)
      X{i} = dlmread (fullfile (edir, sprintf ("cloud__%s__%s.csv", key, names{i})), ",");
    endfor
    ia = find (strcmp (names, anchor));
    refs = [ia; setdiff(find (isref == 1), ia)];
    pooled = [];
    for i = refs' pooled = [pooled; X{i} - median(X{i}, 1)]; endfor
    [~, Sw] = fr_ogk (pooled);
    center = median (X{ia}, 1);
    W = fr_psd_fun (Sw + 1e-12 * eye (columns (Sw)), @(x) 1 ./ sqrt (max (x, 1e-300)));
    np = numel (refs) - 1; ms = {}; Cs = {}; w = [];
    for i = refs'
      [m, C] = fr_fit_gauss ((X{i} - center) * W');
      ms{end+1} = m; Cs{end+1} = C;
      if (i == ia) w(end+1) = ifelse (np == 0, 1, aw); else w(end+1) = (1 - aw) / np; endif
    endfor
    [mb, Cb] = fr_bw_barycenter (ms, Cs, w);
    O = zeros (numel (names), 3);
    for i = 1:numel (names)
      [m, C] = fr_fit_gauss ((X{i} - center) * W');
      l = fr_w2_gauss (m, C, mb, Cb); a = fr_airm (Cb, C);
      O(i, :) = [l a l + a^2];
    endfor
    e = max (abs (O(:) - J(:)) ./ max (abs (J(:)), 1e-9));
    printf ("%-28s d=%d reciters=%d  max rel.err %.2e\n", key, columns (W), numel (names), e);
    worst = max (worst, e);
  endfor
  printf ("WORST %.2e over %d keys -> %s\n", worst, numel (files), ifelse (worst < 1e-6, "AGREE", "DISAGREE"));
  if (worst >= 1e-6) exit (1); endif
endfunction
function r = ifelse (c, a, b)
  if (c) r = a; else r = b; endif
endfunction
