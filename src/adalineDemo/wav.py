"""Float32 signal -> 16-bit PCM WAV bytes, standard library only (the `wave` module)."""

import io
import wave

import numpy as np


def encode(samples: np.ndarray, fs: int) -> bytes:
    """`samples` is float32, roughly in [-1, 1] (values outside are clipped, not rescaled: a
    louder run should sound louder, not be silently normalised back to the same volume)."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(pcm.tobytes())
    return buf.getvalue()
