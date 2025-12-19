import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def spectral_subtraction(noisy_audio, sample_rate, noise_profile_duration=0.5):

    noise_samples = int(noise_profile_duration * sample_rate) #Количество сэмплов, то есть точек замера громкости звука
    noise_samples = min(noise_samples, len(noisy_audio) // 4)

    if noise_samples < 256:
        noise_samples = min(256, len(noisy_audio))

    noise_segment = noisy_audio[:noise_samples]

    fft_size = 2048
    hop_size = fft_size // 4

    # Оконная функция (Ханна) для сглаживания скачков на стыке окон
    window = np.hanning(fft_size)

    # Вычисляем спектр шума
    noise_spectrum = np.zeros(fft_size // 2 + 1)
    num_noise_frames = max(1, len(noise_segment) // hop_size)

    for i in range(num_noise_frames):
        start = i * hop_size
        end = start + fft_size

        if end <= len(noise_segment):
            frame = noise_segment[start:end] * window
            spectrum = np.abs(fft(frame))[:fft_size // 2 + 1]
            noise_spectrum += spectrum

    if num_noise_frames > 0:
        noise_spectrum /= num_noise_frames

    # Применяем спектральное вычитание ко всему сигналу
    output_signal = np.zeros_like(noisy_audio)
    num_frames = (len(noisy_audio) - fft_size) // hop_size + 1

    # Параметры подавления
    alpha = 3.0
    beta = 0.01

    for i in range(num_frames):
        start = i * hop_size
        end = start + fft_size

        frame = noisy_audio[start:end] * window

        # Вычисляем БПФ
        spectrum = fft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Разделяем спектр на положительные и отрицательные частоты
        positive_freqs = magnitude[:fft_size // 2 + 1]

        cleaned_magnitude = np.maximum(positive_freqs - alpha * noise_spectrum, beta * positive_freqs) # Спектральное вычитание шума

        full_cleaned_magnitude = np.zeros_like(magnitude)
        full_cleaned_magnitude[:fft_size // 2 + 1] = cleaned_magnitude
        full_cleaned_magnitude[fft_size // 2 + 1:] = cleaned_magnitude[-2:0:-1]

        cleaned_spectrum = full_cleaned_magnitude * np.exp(1j * phase)
        cleaned_frame = np.real(ifft(cleaned_spectrum))

        output_signal[start:end] += cleaned_frame * window

    if np.max(np.abs(output_signal)) > 0:
        output_signal = output_signal / np.max(np.abs(output_signal)) * np.max(np.abs(noisy_audio))

    return output_signal


def visualize_spectrum(audio, sample_rate, title):
    fft_size = 2048
    spectrum = np.abs(fft(audio[:fft_size]))
    freqs = fftfreq(fft_size, 1 / sample_rate)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:fft_size // 2], 20 * np.log10(spectrum[:fft_size // 2] + 1e-10))
    plt.title(title)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)
    plt.tight_layout()


def main():
    input_file = 'test_waV.wav'
    output_file = 'cleaned_audio.wav'

    try:

        data, fs = sf.read(input_file)


        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        print(f"Загружен файл: {input_file}")
        print(f"Частота дискретизации: {fs} Гц")
        print(f"Длительность: {len(data) / fs:.2f} секунд")


        visualize_spectrum(data, fs, "Спектр исходного сигнала")

        print("Применяем шумоподавление...")
        cleaned_audio = spectral_subtraction(data, fs)

        visualize_spectrum(cleaned_audio, fs, "Спектр после шумоподавления")

        sf.write(output_file, cleaned_audio, fs)
        print(f"Очищенный аудиофайл сохранен как: {output_file}")

        print("\nВоспроизведение оригинального аудио...")
        sd.play(data, fs)
        sd.wait()

        print("Воспроизведение очищенного аудио...")
        sd.play(cleaned_audio, fs)
        sd.wait()

        plt.show()

    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден!")
        print("Создадим тестовый сигнал с шумом для демонстрации...")



if __name__ == "__main__":
    main()