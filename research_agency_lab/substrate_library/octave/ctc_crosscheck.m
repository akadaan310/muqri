function ctc_crosscheck(xdir)
  % Recompute CTC log P and the Viterbi alignment of the clips exported by julia/ctc_run.jl --export
  % and compare with Julia: relative error of log P and of the Viterbi score, and the share of
  % character frames that differ.
  %   octave-cli --path research_agency_lab/substrate_library/octave --eval "ctc_crosscheck('DIR')"
  blank = dlmread(fullfile(xdir, 'blank.tsv'));
  worst = 0; frames_diff = 0; frames_all = 0; k = 0;
  while exist(fullfile(xdir, sprintf('clip%d_lp.tsv', k + 1)), 'file')
    k = k + 1;
    b = fullfile(xdir, sprintf('clip%d', k));
    lp = dlmread([b '_lp.tsv'], '\t');
    seq = dlmread([b '_seq.tsv'], '\t');
    jl = strsplit(strtrim(fileread([b '_jl.tsv'])), "\n");
    ref = str2double(strsplit(jl{1}, "\t"));
    jf = str2double(strsplit(jl{2}, "\t")); jlast = str2double(strsplit(jl{3}, "\t"));
    ll = ctc_forward(lp, seq, blank);
    [sc, f, l] = ctc_viterbi(lp, seq, blank);
    e = max(abs(ll - ref(1)) / abs(ref(1)), abs(sc - ref(2)) / abs(ref(2)));
    d = sum(f ~= jf) + sum(l ~= jlast);
    printf('clip%d  T=%d L=%d  logP %.6f vs %.6f  viterbi %.6f vs %.6f  rel.err %.2e  frame diffs %d/%d\n', ...
           k, rows(lp), numel(seq), ll, ref(1), sc, ref(2), e, d, 2 * numel(seq));
    worst = max(worst, e); frames_diff += d; frames_all += 2 * numel(seq);
  end
  verdict = 'AGREE';
  if worst > 1e-4 || frames_diff > 0.01 * frames_all, verdict = 'DISAGREE'; end
  printf('%s: %d clips, worst rel.err %.2e, frame diffs %d/%d\n', verdict, k, worst, frames_diff, frames_all);
end
