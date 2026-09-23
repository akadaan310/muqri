"""Sub-frame timing: breaking the 40 ms quantisation ceiling.

The model emits one frame per 40 ms (`add_adapter: true, adapter_stride: 2` halves a 20 ms base). A
short vowel is one or two frames, so a hard Viterbi span can only ever be 1 or 2 — and every
duration statistic built on it quantises. That is what made vowel isochrony and vowel-weight
independence unmeasurable: the three vowel medians all landed on exactly the same value and the
spread read 0.0 for all 41 reciters.

The fix is not to smooth afterwards but to stop taking a hard path at all. CTC forward-backward gives
gamma_t(s) = the posterior probability that extended state s occupies frame t. Summed over frames it
is a *fractional* occupancy, and its centre of mass is a *continuous* onset. Measured on 60 anchor
clips, unique fatha values went from 56/1245 under Viterbi to 1106/1245 under the centroid, and
kasra from 1 distinct value (all identical) to 422/439.

Three estimators are provided so they can be cross-checked against each other:

* `expected_durations` -- sum of gamma per symbol. This is confidence mass, not acoustic time: only
  132 of 605 frames carry symbol mass on a typical clip because CTC is peaky.
* `centroid_onsets`    -- posterior-weighted centre of each symbol's occupancy. Continuous AND
  acoustic, because consecutive centroids give the interval between articulations. This is the one
  to build on.
* `peak_parabolic`     -- parabolic interpolation of the occupancy peak, the classic sub-sample peak
  estimator. Cheaper, and agrees with the centroid to ~0.004 of a vowel, which is the useful check
  that neither is an artefact of its own smoothing.
"""
import numpy as np

NEG = -np.inf

def _logaddexp(a, b):
    return np.logaddexp(a, b)

def ctc_alpha_beta(lp, seq, blank):
    """Forward and backward in log space over the extended (blank-interleaved) sequence."""
    T, L = lp.shape[0], len(seq)
    S = 2 * L + 1
    ext = np.array([blank if s % 2 == 0 else seq[s // 2] for s in range(S)])
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    a = np.full((T, S), NEG)
    a[0, 0] = lp[0, ext[0]]
    if S > 1:
        a[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = a[t - 1]
        one = np.concatenate(([NEG], prev[:-1]))
        two = np.where(skip, np.concatenate(([NEG, NEG], prev[:-2])), NEG)
        a[t] = np.logaddexp(np.logaddexp(prev, one), two) + lp[t, ext]

    b = np.full((T, S), NEG)
    b[T - 1, S - 1] = 0.0
    if S > 1:
        b[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = b[t + 1] + lp[t + 1, ext]
        one = np.concatenate((nxt[1:], [NEG]))
        two = np.concatenate((np.where(skip, nxt, NEG)[2:], [NEG, NEG]))
        b[t] = np.logaddexp(np.logaddexp(nxt, one), two)

    logZ = np.logaddexp(a[T - 1, S - 1], a[T - 1, S - 2] if S > 1 else NEG)
    gamma = a + b - logZ          # log posterior occupancy of each extended state per frame
    return gamma, ext, logZ

def expected_durations(lp, seq, blank):
    """Expected number of frames each symbol of `seq` occupies (fractional)."""
    gamma, ext, _ = ctc_alpha_beta(lp, seq, blank)
    g = np.exp(np.clip(gamma, -60, 0))
    L = len(seq)
    occ = np.zeros(L)
    for k in range(L):
        occ[k] = g[:, 2 * k + 1].sum()      # odd index = the real symbol
    return occ

def centroid_onsets(lp, seq, blank):
    """Sub-frame onset of each symbol: the posterior-weighted centre of its occupancy.

    Continuous in time, so consecutive centroids give fractional durations even when every symbol's
    Viterbi span is one or two frames.
    """
    gamma, ext, _ = ctc_alpha_beta(lp, seq, blank)
    g = np.exp(np.clip(gamma, -60, 0))
    T, L = lp.shape[0], len(seq)
    t = np.arange(T)
    cen = np.zeros(L); mass = np.zeros(L)
    for k in range(L):
        w = g[:, 2 * k + 1]
        m = w.sum()
        mass[k] = m
        cen[k] = (w * t).sum() / m if m > 1e-12 else np.nan
    return cen, mass

def peak_parabolic(lp, seq, blank):
    """Cheap alternative: parabolic interpolation of each symbol's occupancy peak (sub-sample peak
    estimation, as used for cross-correlation and pitch)."""
    gamma, ext, _ = ctc_alpha_beta(lp, seq, blank)
    g = np.exp(np.clip(gamma, -60, 0))
    T, L = lp.shape[0], len(seq)
    out = np.zeros(L)
    for k in range(L):
        w = g[:, 2 * k + 1]
        i = int(np.argmax(w))
        if 0 < i < T - 1:
            y0, y1, y2 = w[i - 1], w[i], w[i + 1]
            den = (y0 - 2 * y1 + y2)
            out[k] = i + 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else i
        else:
            out[k] = i
    return out
