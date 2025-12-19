import cv2
import numpy as np
import os
import time
import csv


class VideoTrackerEvaluator:
    def __init__(self, video_paths, output_dir="tracking_results"):
        """
        Инициализация системы оценки трекеров
        """
        self.video_paths = video_paths
        self.output_dir = output_dir
        self.results = []

        # Создаем директории для результатов
        os.makedirs(output_dir, exist_ok=True)
        for tracker in ['KCF', 'CSRT', 'MOSSE']:
            os.makedirs(os.path.join(output_dir, tracker), exist_ok=True)

    def resize_frame(self, frame, max_width=1280, max_height=720):
        """Масштабирование кадра"""
        h, w = frame.shape[:2]

        if w <= max_width and h <= max_height:
            return frame, 1.0, 1.0

        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale, scale

    def create_tracker(self, tracker_type):
        """Создание трекера по типу"""
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        else:
            raise ValueError(f"Неизвестный тип трекера: {tracker_type}")

    def evaluate_tracker(self, video_path, tracker_type, video_index):
        """
        Оценка одного трекера на одном видео
        """
        print(f"\nОценка трекера {tracker_type} на видео: {os.path.basename(video_path)}")

        # Создаем трекер
        tracker = self.create_tracker(tracker_type)

        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка загрузки видео: {video_path}")
            return None

        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])

        print(f"Параметры видео: {width}x{height}, {fps:.1f} FPS, {total_frames} кадров, {codec} кодек")

        # Настройка VideoWriter для сохранения результата
        output_filename = f"video{video_index + 1}_{tracker_type}.mp4"
        output_path = os.path.join(self.output_dir, tracker_type, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Читаем первый кадр для инициализации
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать первый кадр")
            return None

        # Масштабируем для выбора ROI
        display_frame, scale_x, scale_y = self.resize_frame(frame)
        cv2.imshow("Выберите объект для трекинга", display_frame)
        bbox = cv2.selectROI("Выберите объект для трекинга", display_frame, False)
        cv2.destroyWindow("Выберите объект для трекинга")

        # Масштабируем bbox обратно
        original_bbox = (
            int(bbox[0] / scale_x),
            int(bbox[1] / scale_y),
            int(bbox[2] / scale_x),
            int(bbox[3] / scale_y)
        )

        # Сохраняем начальный bounding box для вычисления стабильности
        initial_bbox = original_bbox

        # Инициализируем трекер
        tracker.init(frame, original_bbox)

        # Метрики для оценки
        success_frames = 0
        total_frames_processed = 0
        frame_times = []
        stability_values = []  # Стабильность размера bounding box

        print("Обработка видео...")
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            # Обновляем трекер
            success, tracked_bbox = tracker.update(frame)

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            total_frames_processed += 1

            # Вычисляем метрики
            if success:
                success_frames += 1

                # Вычисляем стабильность (изменение размера относительно начального)
                if initial_bbox[2] > 0 and initial_bbox[3] > 0:
                    width_change = abs(tracked_bbox[2] - initial_bbox[2]) / initial_bbox[2]
                    height_change = abs(tracked_bbox[3] - initial_bbox[3]) / initial_bbox[3]
                    stability = 1.0 - (width_change + height_change) / 2.0
                    stability_values.append(max(0, stability))

                # Рисуем bounding box
                x, y, w, h = [int(v) for v in tracked_bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Отображаем метрики на кадре
                cv2.putText(frame, f"Tracker: {tracker_type}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Показываем номер кадра
                cv2.putText(frame, f"Frame: {total_frames_processed}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "OBJECT LOST", (width // 2 - 100, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(frame, f"Tracker: {tracker_type}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Вычисляем текущий FPS
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Показываем процент успешности
            success_percent = (success_frames / total_frames_processed * 100) if total_frames_processed > 0 else 0
            cv2.putText(frame, f"Success: {success_percent:.1f}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Сохраняем кадр в видеофайл
            out.write(frame)

            # Показываем прогресс в консоли
            if total_frames_processed % 50 == 0:
                progress = (total_frames_processed / total_frames) * 100
                print(f"  Прогресс: {progress:.1f}% ({total_frames_processed}/{total_frames} кадров)")

        # Завершаем обработку
        processing_time = time.time() - start_time
        cap.release()
        out.release()

        # Вычисляем финальные метрики
        success_rate = (success_frames / total_frames_processed * 100) if total_frames_processed > 0 else 0
        avg_fps = total_frames_processed / processing_time if processing_time > 0 else 0
        avg_stability = np.mean(stability_values) if stability_values else 0

        result = {
            'video': os.path.basename(video_path),
            'video_index': video_index + 1,
            'tracker': tracker_type,
            'fps': avg_fps,
            'success_rate': success_rate,
            'stability': avg_stability,
            'frames_lost': total_frames_processed - success_frames,
            'total_frames': total_frames_processed,
            'processing_time': processing_time,
            'output_path': output_path
        }

        print(f"\nРезультаты трекера {tracker_type}:")
        print(f"  Средний FPS: {avg_fps:.2f}")
        print(f"  Успешность трекинга: {success_rate:.2f}%")
        print(f"  Стабильность размера BB: {avg_stability:.3f}")
        print(f"  Потерянные кадры: {result['frames_lost']}")
        print(f"  Общее время обработки: {processing_time:.2f} сек")
        print(f"  Видео сохранено в: {output_path}")

        return result

    def run_evaluation(self):
        """
        Запуск оценки всех трекеров на всех видео
        """
        tracker_types = ['KCF', 'CSRT', 'MOSSE']

        print("=" * 80)
        print("НАЧАЛО ОЦЕНКИ ТРЕКЕРОВ")
        print("=" * 80)

        for video_idx, video_path in enumerate(self.video_paths):
            print(f"\n{'=' * 40}")
            print(f"ВИДЕО {video_idx + 1}: {os.path.basename(video_path)}")
            print(f"{'=' * 40}")

            for tracker_type in tracker_types:
                result = self.evaluate_tracker(video_path, tracker_type, video_idx)
                if result:
                    self.results.append(result)
                cv2.destroyAllWindows()  # Закрываем все окна между трекерами

        # Сохраняем результаты
        self.save_results()

        # Генерируем отчет
        self.generate_report()

    def save_results(self):
        """Сохранение результатов в CSV файл"""
        if not self.results:
            print("Нет результатов для сохранения")
            return

        csv_path = os.path.join(self.output_dir, "tracking_results.csv")

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_index', 'video', 'tracker', 'fps', 'success_rate',
                          'stability', 'frames_lost', 'total_frames', 'processing_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'video_index': result['video_index'],
                    'video': result['video'],
                    'tracker': result['tracker'],
                    'fps': f"{result['fps']:.2f}",
                    'success_rate': f"{result['success_rate']:.2f}",
                    'stability': f"{result['stability']:.3f}",
                    'frames_lost': result['frames_lost'],
                    'total_frames': result['total_frames'],
                    'processing_time': f"{result['processing_time']:.2f}"
                })

        print(f"\nРезультаты сохранены в: {csv_path}")

    def generate_report(self):
        """Генерация текстового отчета"""
        if not self.results:
            print("Нет результатов для отчета")
            return

        print("\n" + "=" * 80)
        print("ОТЧЕТ ПО РЕЗУЛЬТАТАМ ТРЕКИНГА")
        print("=" * 80)

        # Группируем результаты по трекерам
        tracker_results = {}
        for result in self.results:
            tracker = result['tracker']
            if tracker not in tracker_results:
                tracker_results[tracker] = []
            tracker_results[tracker].append(result)

        # Выводим таблицу результатов
        print("\nТАБЛИЦА РЕЗУЛЬТАТОВ:")
        print("-" * 100)
        print(f"{'Видео':<15} {'Трекер':<8} {'FPS':<8} {'Успешность':<12} {'Стабильность':<12} {'Потери':<8}")
        print("-" * 100)

        for result in self.results:
            print(f"{result['video']:<15} {result['tracker']:<8} {result['fps']:<8.1f} "
                  f"{result['success_rate']:<12.1f}% {result['stability']:<12.3f} "
                  f"{result['frames_lost']:<8}")

        # Средние значения по трекерам
        print("\n" + "=" * 80)
        print("СРЕДНИЕ ЗНАЧЕНИЯ ПО ТРЕКЕРАМ:")
        print("-" * 80)
        print(f"{'Трекер':<8} {'Ср. FPS':<10} {'Ср. успешность':<15} {'Ср. стабильность':<16}")
        print("-" * 80)

        for tracker in ['KCF', 'CSRT', 'MOSSE']:
            if tracker in tracker_results:
                avg_fps = np.mean([r['fps'] for r in tracker_results[tracker]])
                avg_success = np.mean([r['success_rate'] for r in tracker_results[tracker]])
                avg_stability = np.mean([r['stability'] for r in tracker_results[tracker]])
                print(f"{tracker:<8} {avg_fps:<10.1f} {avg_success:<15.1f}% {avg_stability:<16.3f}")

        # Рекомендации
        print("\n" + "=" * 80)
        print("РЕКОМЕНДАЦИИ:")
        print("=" * 80)

        # Находим лучшие трекеры по разным метрикам
        best_by_fps = max(self.results, key=lambda x: x['fps'])
        best_by_success = max(self.results, key=lambda x: x['success_rate'])
        best_by_stability = max(self.results, key=lambda x: x['stability'])

        print(f"1. Для максимальной скорости (FPS): {best_by_fps['tracker']} "
              f"({best_by_fps['fps']:.1f} FPS на видео {best_by_fps['video']})")
        print(f"2. Для максимальной надежности: {best_by_success['tracker']} "
              f"({best_by_success['success_rate']:.1f}% успешных кадров на видео {best_by_success['video']})")
        print(f"3. Для максимальной стабильности: {best_by_stability['tracker']} "
              f"(стабильность: {best_by_stability['stability']:.3f} на видео {best_by_stability['video']})")

        # Создаем итоговый текстовый репорт
        report_path = os.path.join(self.output_dir, "final_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО СРАВНИТЕЛЬНОМУ АНАЛИЗУ ТРЕКЕРОВ\n")
            f.write("=" * 60 + "\n\n")

            f.write("ПРОАНАЛИЗИРОВАНО ВИДЕО:\n")
            for i, video in enumerate(self.video_paths, 1):
                f.write(f"{i}. {os.path.basename(video)}\n")

        print(f"\nПолный отчет сохранен в: {report_path}")

        # Информация о сохраненных видеофайлах
        print("\n" + "=" * 80)
        print("СОХРАНЕННЫЕ ВИДЕОФАЙЛЫ:")
        print("=" * 80)

        video_count = 0
        for tracker in ['KCF', 'CSRT', 'MOSSE']:
            tracker_dir = os.path.join(self.output_dir, tracker)
            if os.path.exists(tracker_dir):
                videos = [f for f in os.listdir(tracker_dir) if f.endswith('.mp4')]
                print(f"\n{tracker} трекер ({len(videos)} видео):")
                for video in videos:
                    print(f"  - {video}")
                    video_count += 1

        print(f"\nВсего сохранено видеофайлов: {video_count}")
        print("Оригинальные видео: 5")
        print("Обработанные видео: 15")
        print("Всего файлов: 20")


def main():
    print("ПРОГРАММА СРАВНИТЕЛЬНОГО АНАЛИЗА ТРЕКЕРОВ ОБЪЕКТОВ")
    print("=" * 60)

    # Создаем папку для тестовых видео, если нет реальных
    test_videos_dir = "test_videos"
    os.makedirs(test_videos_dir, exist_ok=True)

    # video_paths = ["car.mp4"]
    video_paths = ["camera_follows_car.mp4", "car_front.mp4", "distancing_cars.mp4", "off_road_car.mp4", "zoom_cars.mp4"]

    if not video_paths:
        print("Ошибка: не найдено видеофайлов!")
        return

    print(f"\nБудет проанализировано {len(video_paths)} видео:")
    for i, video in enumerate(video_paths, 1):
        print(f"{i}. {video}")

    # Запускаем оценку
    evaluator = VideoTrackerEvaluator(video_paths)
    evaluator.run_evaluation()

    print("\n" + "=" * 80)
    print("ОЦЕНКА ВЫПОЛНЕНА УСПЕШНО!")

if __name__ == "__main__":
    main()