function qaari_features(jobs_csv, out_csv)
% QAARI_FEATURES  Independent DSP measurements of Tajweed diagnostic spans (GNU Octave + signal).
%
%   qaari_features('jobs.csv', 'features.csv')
%
% jobs.csv columns (header required):  span_id,wav_path,start_s,end_s
% Spans of the same wav should be contiguous rows (each file is read once).
%
% For every span the engine measures, independently of the Python/Praat path:
%   core_ms       vowel/sonorant core: longest run where the Hilbert envelope of the 100-1000 Hz
%                 band stays within 10 dB of the span peak AND the frame is voiced. This removes
%                 the consonant closures that CTC spans absorb (the harakah inflation problem).
%   voiced_frac   fraction of frames whose normalised autocorrelation peak (70-400 Hz) > 0.45
%   f0_hz         median cepstral F0 (quefrency peak near the autocorrelation lag) over voiced frames
%   hnr_db        autocorrelation HNR (Boersma 1993): 10*log10(r/(1-r)) at the pitch lag, median
%   f1_hz f2_hz f3_hz  LPC formants, order P = 2 + fs/1000, at the core midpoint (pole angles with
%                 bandwidth < 400 Hz, sorted), averaged over the middle 40 % of the core
%   nasal_db      low-band (0-400 Hz) minus mid-band (400-2500 Hz) energy, dB (murmur contrast)
%   burst_db      peak positive spectral-flux rise in the span, dB (qalqalah release)
%   hf_ratio      share of energy above 2.5 kHz (frication: safir / tafashhi)

  pkg load signal
  fid = fopen(jobs_csv, 'r');
  header = fgetl(fid); %#ok<NASGU>
  C = textscan(fid, '%s %s %f %f', 'Delimiter', ',');
  fclose(fid);
  ids = C{1}; wavs = C{2}; t0s = C{3}; t1s = C{4};
  n = numel(ids);

  out = fopen(out_csv, 'w');
  fprintf(out, 'span_id,core_ms,voiced_frac,f0_hz,hnr_db,f1_hz,f2_hz,f3_hz,nasal_db,burst_db,hf_ratio\n');
  cur = ''; x = []; fs = 16000;
  for k = 1:n
    if ~strcmp(wavs{k}, cur)
      cur = wavs{k};
      [x, fs] = audioread(cur);
      x = mean(x, 2);
      if fs ~= 16000
        x = resample(x, 16000, fs); fs = 16000;
      end
      x = x - mean(x);
    end
    a = max(1, floor(t0s(k) * fs) + 1); b = min(numel(x), floor(t1s(k) * fs));
    if b - a < round(0.03 * fs)
      fprintf(out, '%s,,,,,,,,,,\n', ids{k}); continue;
    end
    r = span_features(x(a:b), fs);
    fprintf(out, '%s,%.2f,%.3f,%.1f,%.2f,%.0f,%.0f,%.0f,%.2f,%.2f,%.3f\n', ids{k}, r.core_ms, r.voiced_frac, ...
            r.f0, r.hnr, r.f1, r.f2, r.f3, r.nasal_db, r.burst_db, r.hf_ratio);
  end
  fclose(out);
end

function r = span_features(s, fs)
  hop = round(0.005 * fs); win = round(0.03 * fs);
  nf = max(1, floor((numel(s) - win) / hop) + 1);
  % --- Hilbert envelope of the vowel band ------------------------------------------------------
  [bb, aa] = butter(4, [100 1000] / (fs / 2));
  sv = filtfilt(bb, aa, s);
  env = abs(hilbert(sv));
  env = filter(ones(round(0.02 * fs), 1) / round(0.02 * fs), 1, env);
  env_db = 20 * log10(env + 1e-9);
  % --- frame-wise voicing / F0 (cepstrum) and HNR (autocorrelation) ------------------------------
  voiced = false(nf, 1); f0 = nan(nf, 1); hnr = nan(nf, 1);
  w = hanning(win);
  qlo = round(fs / 400); qhi = round(fs / 70);
  for i = 1:nf
    fr = s((i - 1) * hop + (1:win)) .* w;
    if ~any(fr), continue; end
    % Voicing + HNR: normalised autocorrelation peak in the 70-400 Hz lag range (window-corrected,
    % Boersma 1993); voiced when r > 0.45 (Praat's default voicing threshold).
    ac = xcorr(fr, 'coeff'); ac = ac(win:end);
    acw = xcorr(w, 'coeff'); acw = acw(win:end);
    acn = ac ./ max(acw, 1e-3);
    [rr, ix] = max(acn(qlo + 1:min(qhi + 1, numel(acn))));
    if rr > 0.45
      voiced(i) = true;
      rr = min(rr, 0.9999);
      hnr(i) = 10 * log10(rr / (1 - rr));
      % F0 from the real cepstrum peak (quefrency), refined around the autocorrelation lag.
      ceps = real(ifft(log(abs(fft(fr, 1024)) + 1e-9)));
      lag0 = qlo + ix - 1;
      lo = max(qlo, round(lag0 * 0.8)); hi = min(qhi, round(lag0 * 1.2));
      [~, j] = max(ceps(lo + 1:hi + 1));
      f0(i) = fs / (lo + j - 1);
    end
  end
  % --- vowel core: envelope within 10 dB of peak AND voiced, longest run -----------------------
  fr_env = env_db(min(numel(env_db), (0:nf - 1)' * hop + round(win / 2)));
  good = (fr_env > max(fr_env) - 10) & voiced;
  [run_len, run_start] = longest_run(good);
  r.core_ms = run_len * hop / fs * 1000;
  r.voiced_frac = mean(voiced);
  r.f0 = safe_median(f0(voiced));
  r.hnr = safe_median(hnr(~isnan(hnr)));
  % --- LPC formants over the middle of the core ----------------------------------------------
  P = 2 + round(fs / 1000);
  F = nan(0, 3);
  if run_len >= 4
    mids = run_start + round(run_len * 0.3) : run_start + round(run_len * 0.7);
    pre = filter([1 -0.97], 1, s);
    for i = mids
      fr = pre((i - 1) * hop + (1:win)) .* w;
      if any(fr)
        A = lpc(fr, P);
        z = roots(A); z = z(imag(z) > 0);
        fq = angle(z) * fs / (2 * pi); bw = -log(abs(z)) * fs / pi;
        fq = sort(fq(bw < 400 & fq > 90));
        if numel(fq) >= 3, F(end + 1, :) = fq(1:3)'; end %#ok<AGROW>
      end
    end
  end
  if isempty(F), r.f1 = NaN; r.f2 = NaN; r.f3 = NaN;
  else, m = median(F, 1); r.f1 = m(1); r.f2 = m(2); r.f3 = m(3); end
  % --- nasal murmur contrast, spectral-flux burst, high-frequency share ----------------------
  S = abs(fft(s .* hanning(numel(s)), 2 ^ nextpow2(numel(s)))) .^ 2;
  fq = (0:numel(S) - 1)' * fs / numel(S); half = fq <= fs / 2;
  S = S(half); fq = fq(half);
  r.nasal_db = 10 * log10(sum(S(fq < 400)) + 1e-12) - 10 * log10(sum(S(fq >= 400 & fq < 2500)) + 1e-12);
  r.hf_ratio = sum(S(fq >= 2500)) / (sum(S) + 1e-12);
  flux = zeros(nf, 1); prev = [];
  for i = 1:nf
    X = abs(fft(s((i - 1) * hop + (1:win)) .* w, 512)); X = 20 * log10(X(1:257) + 1e-9);
    if ~isempty(prev), flux(i) = mean(max(X - prev, 0)); end
    prev = X;
  end
  r.burst_db = max(flux);
end

function m = safe_median(v)
  if isempty(v), m = NaN; else, m = median(v); end
end

function [best, best_start] = longest_run(v)
  best = 0; best_start = 1; run = 0;
  for i = 1:numel(v)
    if v(i), run = run + 1; else, run = 0; end
    if run > best, best = run; best_start = i - run + 1; end
  end
end
