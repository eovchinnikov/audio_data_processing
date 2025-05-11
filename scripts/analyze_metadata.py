import pandas as pd
from pathlib import Path
import time
from shutil import copy2, rmtree

import librosa
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def analyze_metadata(tsv_path, clips_dir, valid_clips_dir):
    # Загружаем validated.tsv
    df = pd.read_csv(tsv_path, sep='\t')

    # Фильтруем только записи с указанием мужского или женского пола
    df = df[df['gender'].isin(['male_masculine', 'female_feminine'])]
    df['gender'] = df['gender'].map({
        'male_masculine': 'male',
        'female_feminine': 'female'
    })

    # Добавляем полный путь до аудиофайлов
    df['file_path'] = df['path'].apply(lambda x: str(Path(clips_dir) / x))

    durations = []
    valid_sample_rates = []
    vad_flags = []

    vad_model = load_silero_vad()
    start_time = time.time()

    target_hours_per_gender = 25
    target_seconds_per_gender = target_hours_per_gender * 3600

    male_seconds = 0
    female_seconds = 0

    for idx, (path, gender) in enumerate(zip(df['file_path'], df['gender'])):
        if male_seconds >= target_seconds_per_gender and female_seconds >= target_seconds_per_gender:
            elapsed = time.time() - start_time
            print(f"Обработано файлов: {len(durations)}, "
                  f"Время: {elapsed:.2f} сек, "
                  f"Мужской голос: {male_seconds / 3600:.2f} ч, "
                  f"Женский голос: {female_seconds / 3600:.2f} ч")
            print("Достигнут таргет объём данных по обоим полам.")
            break

        try:
            y, sr = librosa.load(path, sr=None)
            duration = len(y) / sr

            if (gender == 'male' and male_seconds >= target_seconds_per_gender) or \
               (gender == 'female' and female_seconds >= target_seconds_per_gender):
                durations.append(0)
                valid_sample_rates.append(False)
                vad_flags.append(False)
            elif sr < 16000 or duration <= 1:
                durations.append(0)
                valid_sample_rates.append(False)
                vad_flags.append(False)
            else:
                wav = read_audio(path, sampling_rate=16000)
                speech_timestamps = get_speech_timestamps(wav, vad_model, return_seconds=True)
                speech_detected = len(speech_timestamps) > 0

                if not speech_detected:
                    durations.append(0)
                    valid_sample_rates.append(True)
                    vad_flags.append(False)
                else:
                    durations.append(duration)
                    valid_sample_rates.append(True)
                    vad_flags.append(True)

                    # Копируем валидный файл в новую директорию
                    valid_file_path = valid_clips_dir / Path(path).name
                    copy2(path, valid_file_path)

                    # Обновляем путь в датафрейме
                    df.loc[idx, 'file_path'] = str(valid_file_path)

                    if gender == 'male':
                        male_seconds += duration
                    elif gender == 'female':
                        female_seconds += duration

        except Exception as e:
            print(f"Ошибка при чтении файла {path}: {e}")
            durations.append(0)
            valid_sample_rates.append(False)
            vad_flags.append(False)

        if len(durations) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Обработано файлов: {len(durations)}, "
                  f"Время: {elapsed:.2f} сек, "
                  f"Мужской голос: {male_seconds / 3600:.2f} ч, "
                  f"Женский голос: {female_seconds / 3600:.2f} ч")

    remaining_rows = len(df) - len(durations)
    durations.extend([0] * remaining_rows)
    valid_sample_rates.extend([False] * remaining_rows)
    vad_flags.extend([False] * remaining_rows)

    df['duration_sec'] = durations
    df['valid_sample_rate'] = valid_sample_rates
    df['vad_speech_detected'] = vad_flags

    # Оставляем только валидные данные
    df = df[(df['valid_sample_rate']) & (df['vad_speech_detected'])]

    # Оставляем нужные колонки
    final_columns = ['file_path', 'sentence', 'gender', 'duration_sec', 'age']
    final_columns = [col for col in final_columns if col in df.columns]  # Оставляем только существующие
    df = df[final_columns]

    elapsed_total = time.time() - start_time
    print(f"\nФинальное количество файлов: {len(df)}")
    print(f"Общее время выполнения: {elapsed_total:.2f} сек")

    return df


if __name__ == "__main__":
    tsv_path = "data/raw/cv-corpus-20.0-2024-12-06/ru/validated.tsv"
    clips_dir = "data/raw/cv-corpus-20.0-2024-12-06/ru/clips"

    valid_clips_dir = Path("data/processed/valid_clips")
    # Полная очистка папки перед новым запуском
    if valid_clips_dir.exists():
        rmtree(valid_clips_dir)
    valid_clips_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = analyze_metadata(tsv_path, clips_dir, valid_clips_dir)

    output_path = "data/processed/metadata.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)
    print(f"\nИтоговые метаданные сохранены в: {output_path}")
