#!/usr/bin/env python3
"""Lọc phổ một tệp âm thanh theo hồ sơ tần số của tệp tham chiếu."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SAMPLE_RATE = 16_000
N_FFT = 2048
HOP_LENGTH = 256
REFERENCE_PERCENTILE = 75
MASK_LOW_DB = -40.0
MASK_HIGH_DB = -10.0
MASK_FLOOR_GAIN = 0.02


def load_audio(path: Path) -> np.ndarray:
    """Đọc mono và resample về tần số chuẩn của model audio."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f"Tệp âm thanh rỗng: {path}")
    return audio.astype(np.float32, copy=False)


def stft(audio: np.ndarray) -> np.ndarray:
    """Tính STFT phức với cùng cấu hình cho nguồn và tham chiếu."""
    return librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
    )


def spectral_profile(stft_matrix: np.ndarray) -> np.ndarray:
    """Lấy percentile theo thời gian để giữ đặc trưng của các tiếng gõ ngắn."""
    return np.percentile(
        np.abs(stft_matrix),
        REFERENCE_PERCENTILE,
        axis=1,
    )


def build_reference_mask(profile: np.ndarray) -> np.ndarray:
    """Tạo mặt nạ mềm, giảm mạnh các bin không nổi bật trong tệp tham chiếu."""
    peak = max(float(profile.max()), 1e-12)
    profile_db = 20.0 * np.log10(np.maximum(profile / peak, 1e-12))
    mask = np.clip(
        (profile_db - MASK_LOW_DB) / (MASK_HIGH_DB - MASK_LOW_DB),
        0.0,
        1.0,
    )
    mask = gaussian_filter1d(mask, sigma=2.0)
    return MASK_FLOOR_GAIN + (1.0 - MASK_FLOOR_GAIN) * mask


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Đo độ giống nhau giữa hai hồ sơ phổ trên thang từ 0 đến 1."""
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def energy_band(profile: np.ndarray, frequencies: np.ndarray) -> tuple[float, float]:
    """Tìm dải liên tục chứa 90 phần trăm năng lượng phổ tham chiếu."""
    power = np.square(profile)
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    cumulative = np.cumsum(power) / total
    low_index = int(np.searchsorted(cumulative, 0.05))
    high_index = min(int(np.searchsorted(cumulative, 0.95)), frequencies.size - 1)
    return float(frequencies[low_index]), float(frequencies[high_index])


def dominant_frequencies(profile: np.ndarray, frequencies: np.ndarray) -> list[float]:
    """Lấy tối đa năm đỉnh phổ mạnh và tách biệt của tệp tham chiếu."""
    peaks, _ = find_peaks(profile, distance=12)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(profile))])
    strongest = peaks[np.argsort(profile[peaks])[-5:]][::-1]
    return [float(frequencies[index]) for index in strongest]


def db_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """Đổi biên độ STFT sang dB để vẽ phổ thời gian-tần số."""
    return librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)


def draw_visualization(
    target: np.ndarray,
    reference: np.ndarray,
    filtered: np.ndarray,
    target_stft: np.ndarray,
    reference_stft: np.ndarray,
    filtered_stft: np.ndarray,
    frequencies: np.ndarray,
    target_profile: np.ndarray,
    reference_profile: np.ndarray,
    filtered_profile: np.ndarray,
    output_path: Path,
) -> None:
    """Vẽ waveform, spectrogram và phổ trung bình trước/sau lọc."""
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)

    target_time = np.arange(target.size) / SAMPLE_RATE
    reference_time = np.arange(reference.size) / SAMPLE_RATE
    filtered_time = np.arange(filtered.size) / SAMPLE_RATE
    axes[0, 0].plot(target_time, target, linewidth=0.7)
    axes[0, 0].set_title("Target waveform")
    axes[0, 1].plot(reference_time, reference, linewidth=0.7, color="tab:orange")
    axes[0, 1].set_title("Reference waveform")

    for axis, matrix, title in (
        (axes[1, 0], target_stft, "Target spectrogram"),
        (axes[1, 1], reference_stft, "Reference spectrogram"),
        (axes[2, 0], filtered_stft, "Filtered target spectrogram"),
    ):
        librosa.display.specshow(
            db_spectrogram(matrix),
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=axis,
            cmap="magma",
        )
        axis.set_title(title)
        axis.set_ylim(0, SAMPLE_RATE / 2)

    epsilon = 1e-12
    axes[2, 1].plot(
        frequencies,
        20 * np.log10(target_profile / max(float(target_profile.max()), epsilon) + epsilon),
        label="Target",
        alpha=0.8,
    )
    axes[2, 1].plot(
        frequencies,
        20 * np.log10(
            reference_profile / max(float(reference_profile.max()), epsilon) + epsilon
        ),
        label="Reference",
        alpha=0.8,
    )
    axes[2, 1].plot(
        frequencies,
        20 * np.log10(
            filtered_profile / max(float(filtered_profile.max()), epsilon) + epsilon
        ),
        label="Filtered",
        alpha=0.8,
    )
    axes[2, 1].set_title("Average spectral profiles")
    axes[2, 1].set_xlim(0, SAMPLE_RATE / 2)
    axes[2, 1].set_ylim(-80, 5)
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_ylabel("Relative magnitude (dB)")
    axes[2, 1].grid(alpha=0.25)
    axes[2, 1].legend()

    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 1].set_xlabel("Time (s)")
    figure.suptitle("Frequency comparison and reference-profile filtering", fontsize=16)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def process(
    target_path: Path,
    reference_path: Path,
    output_audio_path: Path,
    output_plot_path: Path,
) -> None:
    """So sánh phổ, lọc nguồn và ghi toàn bộ kết quả."""
    target = load_audio(target_path)
    reference = load_audio(reference_path)
    target_stft = stft(target)
    reference_stft = stft(reference)
    target_profile = spectral_profile(target_stft)
    reference_profile = spectral_profile(reference_stft)
    reference_mask = build_reference_mask(reference_profile)

    filtered_stft = target_stft * reference_mask[:, np.newaxis]
    filtered = librosa.istft(
        filtered_stft,
        hop_length=HOP_LENGTH,
        length=target.size,
    ).astype(np.float32)
    filtered_profile = spectral_profile(filtered_stft)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio_path, filtered, SAMPLE_RATE, subtype="PCM_16")

    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    draw_visualization(
        target,
        reference,
        filtered,
        target_stft,
        reference_stft,
        filtered_stft,
        frequencies,
        target_profile,
        reference_profile,
        filtered_profile,
        output_plot_path,
    )

    low_frequency, high_frequency = energy_band(reference_profile, frequencies)
    dominant = dominant_frequencies(reference_profile, frequencies)
    before_similarity = cosine_similarity(target_profile, reference_profile)
    after_similarity = cosine_similarity(filtered_profile, reference_profile)
    target_rms = float(np.sqrt(np.mean(np.square(target))))
    filtered_rms = float(np.sqrt(np.mean(np.square(filtered))))
    retained_rms = filtered_rms / max(target_rms, 1e-12)

    print(f"target_duration={target.size / SAMPLE_RATE:.3f}")
    print(f"reference_duration={reference.size / SAMPLE_RATE:.3f}")
    print(f"reference_energy_band_90hz={low_frequency:.1f}-{high_frequency:.1f}")
    print("reference_dominant_hz=" + ",".join(f"{value:.1f}" for value in dominant))
    print(f"spectral_similarity_before={before_similarity:.6f}")
    print(f"spectral_similarity_after={after_similarity:.6f}")
    print(f"filtered_rms_ratio={retained_rms:.6f}")
    print(f"output_audio={output_audio_path}")
    print(f"output_plot={output_plot_path}")


def main() -> None:
    """Nhận đường dẫn từ dòng lệnh và chạy bộ lọc tham chiếu."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-audio", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    process(args.target, args.reference, args.output_audio, args.output_plot)


if __name__ == "__main__":
    main()
