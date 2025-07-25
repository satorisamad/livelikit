import io

from pydub import AudioSegment  # type: ignore[import-untyped]


def resample_audio(audio_bytes: bytes, original_sample_rate: int, target_sample_rate: int) -> bytes:
    resampled_audio = AudioSegment.from_raw(
        io.BytesIO(audio_bytes),
        sample_width=2,
        frame_rate=original_sample_rate,
        channels=1,
    ).set_frame_rate(target_sample_rate)
    return resampled_audio.raw_data  # type: ignore
