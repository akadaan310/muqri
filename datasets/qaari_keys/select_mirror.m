function select_mirror(bdir)
  % Octave mirror of select.jl (QaariKeys nested tiers). Same needs, weights, fill rule and tie-break;
  % writes BDIR/tiers_octave.tsv, which must equal BDIR/tiers_julia.tsv line for line.
  %   octave-cli --path datasets/qaari_keys --eval "select_mirror('datasets/qaari_keys/build')"
  ay = dlmread(fullfile(bdir, 'ayahs.tsv'), '\t');
  keys = strsplit(strtrim(fileread(fullfile(bdir, 'keys.txt'))), "\n");
  t = dlmread(fullfile(bdir, 'ayah_keys.tsv'), '\t');
  C = full(sparse(t(:,1), t(:,2), t(:,3), rows(ay), numel(keys)));
  words = ay(:,3);
  fam = cellfun(@(k) strtok(k, ':'), keys, 'UniformOutput', false);
  F = {'rule','letter','pair','rep','special','waqf'};
  % need / weight per family, in the order of F
  tiers = struct( ...
    'name',   {'T10', 'T100', 'T300'}, ...
    'size',   {10, 100, 300}, ...
    'need',   {[1 1 1 1 0 1], [3 3 2 3 1 3], [8 8 5 8 1 8]}, ...
    'weight', {[1 .5 1.5 1 0 .5], [1 .5 1.5 1 2 .5], [1 .5 1.5 1 2 .5]}, ...
    'lambda', {12, 15, 20}, ...
    'force',  {false, true, true});
  [~, fi] = ismember(fam, F);
  total = sum(C, 1);
  isspecial = strcmp(fam, 'special');
  fid = fopen(fullfile(bdir, 'tiers_octave.tsv'), 'w');
  prev = [];
  for ti = 1:numel(tiers)
    T = tiers(ti);
    need = min(T.need(fi), total);
    wf = T.weight(fi);
    w = zeros(size(need)); w(need > 0) = wf(need > 0) ./ need(need > 0);
    chosen = prev(:)';
    got = zeros(1, numel(keys));
    if ~isempty(chosen), got = sum(C(chosen, :), 1); end
    if T.force
      for a = 1:rows(C)
        if any(chosen == a), continue; end
        if any(C(a, isspecial) > 0)
          chosen(end+1) = a; got = got + C(a, :);
        end
      end
    end
    taken = false(rows(C), 1); taken(chosen) = true;
    while numel(chosen) < T.size
      rem = max(0, need - got);
      g = min(C, rem) * w';                       % broadcast: min(c_ak, rem_k)
      v = g ./ (1 + words / T.lambda);
      best = 0; bestv = 0;
      for a = 1:rows(C)
        if taken(a) || g(a) == 0, continue; end
        if v(a) > bestv * (1 + 1e-12), best = a; bestv = v(a); end
      end
      if best == 0
        if ~any(need < total), break; end
        need(need > 0) = min(2 * need(need > 0), total(need > 0));
        w = zeros(size(need)); w(need > 0) = wf(need > 0) ./ need(need > 0);
        continue;
      end
      chosen(end+1) = best; taken(best) = true; got = got + C(best, :);
    end
    fprintf(fid, '%s\t%d\n', [repmat({T.name}, 1, numel(chosen)); num2cell(chosen)]{:});
    printf('%-5s ayahs=%d words=%d\n', T.name, numel(chosen), sum(words(chosen)));
    prev = chosen;
  end
  fclose(fid);
end
