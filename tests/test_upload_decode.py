"""Uploads in the formats people actually send must decode, and junk must be refused cleanly.

The bug this pins: a phone recording named `.wav` is usually AAC in an MP4 container. libsndfile
rejects it, and the old fallback (`librosa.load` on bytes) only retried libsndfile, so the page
showed a LibsndfileError traceback. ffmpeg now handles everything libsndfile cannot, reading from a
temp file because MP4 keeps its index at the end and cannot be read from a pipe.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from app.webapp import AudioDecodeError, decode_upload  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# (ffmpeg output args, what it imitates)
FORMATS = {
    "pcm_wav": (["-c:a", "pcm_s16le", "-f", "wav"], "an ordinary WAV, 44.1 kHz stereo"),
    # an MP4 cannot be written to a pipe unless fragmented; the unfragmented layout is tested below
    "aac_named_wav": (["-c:a", "aac", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov"],
                      "a phone recording: AAC/MP4, saved as .wav"),
    "adpcm_wav": (["-c:a", "adpcm_ms", "-f", "wav"], "a WAV with a compressed codec"),
    "mp3": (["-c:a", "libmp3lame", "-f", "mp3"], "MP3"),
    "opus_webm": (["-c:a", "libopus", "-f", "webm"], "a browser MediaRecorder capture"),
}


def encode(args: list[str]) -> bytes:
    run = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
                          "-i", "sine=frequency=440:duration=2:sample_rate=44100", "-ac", "2",
                          *args, "pipe:1"],
                         capture_output=True, check=False)
    if run.returncode != 0 or not run.stdout:
        pytest.skip(f"this ffmpeg cannot encode {args}: {run.stderr.decode()[-200:]}")
    return run.stdout


@pytest.mark.parametrize("name", list(FORMATS))
def test_common_upload_formats_decode_to_16k_mono(name: str) -> None:
    wave = decode_upload(encode(FORMATS[name][0]))
    assert wave.dtype == np.float32 and wave.ndim == 1
    assert abs(wave.size / 16000 - 2.0) < 0.1, f"{name}: {wave.size / 16000:.2f}s, expected 2 s"
    assert 0.05 < float(np.abs(wave).max()) <= 1.01


def test_mp4_with_index_at_the_end_decodes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The default MP4 layout (moov atom last) is exactly what a pipe cannot read."""
    out = tmp_path / "phone.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "-f", "mp4", str(out)],
                   check=True)
    wave = decode_upload(out.read_bytes())
    assert abs(wave.size / 16000 - 2.0) < 0.1


def test_junk_is_refused_with_a_reason_not_a_traceback() -> None:
    with pytest.raises(AudioDecodeError, match="could not decode"):
        decode_upload(bytes(range(256)) * 200)
