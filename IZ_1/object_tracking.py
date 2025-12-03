# fixed_tracking_final.py
# ПОЛНАЯ РЕАЛИЗАЦИЯ ДЛЯ ИНДИВИДУАЛЬНОГО ЗАДАНИЯ ПО ТРЕКИНГУ

import cv2
import numpy as np
import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

print("=== COMPLETE TRACKING ANALYSIS SYSTEM ===")
print(f"OpenCV version: {cv2.__version__}")


class TrackingAnalyzer:
    """Анализатор качества трекинга"""

    def __init__(self):
        self.metrics = {
            'success_frames': 0,
            'total_frames': 0,
            'tracking_loss_count': 0,
            'recovery_count': 0,
            'center_positions': [],
            'bbox_sizes': [],
            'processing_times': []
        }

    def update(self, success, bbox, processing_time):
        """Обновление метрик"""
        self.metrics['total_frames'] += 1
        self.metrics['processing_times'].append(processing_time)

        if success:
            self.metrics['success_frames'] += 1
            x, y, w, h = bbox
            center = (x + w / 2, y + h / 2)
            self.metrics['center_positions'].append(center)
            self.metrics['bbox_sizes'].append((w, h))

            # Проверяем восстановление после потери
            if len(self.metrics['center_positions']) > 1 and self.metrics['tracking_loss_count'] > 0:
                prev_center = self.metrics['center_positions'][-2]
                distance = np.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
                if distance < 50:  # Порог для определения восстановления
                    self.metrics['recovery_count'] += 1
        else:
            self.metrics['tracking_loss_count'] += 1
            self.metrics['center_positions'].append(None)
            self.metrics['bbox_sizes'].append(None)

    def calculate_metrics(self):
        """Расчет итоговых метрик"""
        total_frames = self.metrics['total_frames']
        success_frames = self.metrics['success_frames']

        metrics = {
            'success_rate': (success_frames / total_frames * 100) if total_frames > 0 else 0,
            'tracking_loss_frequency': (
                        self.metrics['tracking_loss_count'] / total_frames * 100) if total_frames > 0 else 0,
            'recovery_rate': (self.metrics['recovery_count'] / self.metrics['tracking_loss_count'] * 100) if
            self.metrics['tracking_loss_count'] > 0 else 0,
            'average_processing_time': np.mean(self.metrics['processing_times']) if self.metrics[
                'processing_times'] else 0,
            'stability_score': self._calculate_stability(),
            'total_frames_processed': total_frames
        }
        return metrics

    def _calculate_stability(self):
        """Расчет стабильности трекинга на основе изменения размера bbox"""
        if len(self.metrics['bbox_sizes']) < 2:
            return 0

        sizes = [size for size in self.metrics['bbox_sizes'] if size is not None]
        if len(sizes) < 2:
            return 0

        size_changes = []
        for i in range(1, len(sizes)):
            area1 = sizes[i - 1][0] * sizes[i - 1][1]
            area2 = sizes[i][0] * sizes[i][1]
            change = abs(area2 - area1) / area1
            size_changes.append(change)

        stability = 1 - np.mean(size_changes)
        return max(0, min(1, stability)) * 100


class TrackingSystem:
    def __init__(self):
        self.tracker = None
        self.tracker_type = None
        self.trajectory = []
        self.is_initialized = False
        self.analyzer = TrackingAnalyzer()

    def initialize_tracker(self, tracker_type='CSRT'):
        """Инициализация конкретного трекера"""
        try:
            if tracker_type == 'CSRT':
                self.tracker = cv2.legacy.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                self.tracker = cv2.legacy.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                self.tracker = cv2.legacy.TrackerMOSSE_create()
            else:
                print(f"Неизвестный тип трекера: {tracker_type}")
                return False

            self.tracker_type = tracker_type
            print(f"Трекер {tracker_type} создан успешно")
            return True

        except Exception as e:
            print(f"Ошибка создания трекера {tracker_type}: {e}")
            return False

    def init_tracking(self, frame, bbox):
        """Инициализация отслеживания с выбранной областью"""
        if self.tracker is None:
            print("Трекер не инициализирован")
            return False

        try:
            success = self.tracker.init(frame, bbox)
            if success:
                self.is_initialized = True
                self.trajectory = []
                # Добавляем первую точку в траекторию
                x, y, w, h = bbox
                center = (int(x + w / 2), int(y + h / 2))
                self.trajectory.append(center)
                print(f"Трекер {self.tracker_type} инициализирован с bbox: {bbox}")
            else:
                print("Не удалось инициализировать трекер с выбранной областью")
            return success
        except Exception as e:
            print(f"Ошибка инициализации трекера: {e}")
            return False

    def update_tracking(self, frame):
        """Обновление позиции трекера"""
        if not self.is_initialized or self.tracker is None:
            return False, None

        try:
            start_time = time.time()
            success, bbox = self.tracker.update(frame)
            processing_time = time.time() - start_time

            # Обновляем метрики
            self.analyzer.update(success, bbox, processing_time)

            if success:
                # Обновляем траекторию
                x, y, w, h = bbox
                center = (int(x + w / 2), int(y + h / 2))
                self.trajectory.append(center)

                # Ограничиваем длину траектории
                if len(self.trajectory) > 50:
                    self.trajectory.pop(0)

            return success, bbox

        except Exception as e:
            print(f"Ошибка обновления трекера: {e}")
            return False, None


class CustomMeanShiftTracker:
    """Собственная реализация MeanShift трекера"""

    def __init__(self):
        self.roi_hist = None
        self.track_window = None
        self.termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def init(self, frame, bbox):
        """Инициализация трекера"""
        x, y, w, h = [int(v) for v in bbox]
        self.track_window = (x, y, w, h)

        # Выделяем ROI
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Создаем маску и гистограмму
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        return True

    def update(self, frame):
        """Обновление позиции объекта"""
        if self.roi_hist is None:
            return False, None

        try:
            start_time = time.time()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

            # Применяем MeanShift
            ret, self.track_window = cv2.meanShift(dst, self.track_window, self.termination_criteria)

            processing_time = time.time() - start_time

            if ret:
                x, y, w, h = self.track_window
                return True, (x, y, w, h), processing_time
            else:
                return False, None, processing_time

        except Exception as e:
            print(f"Ошибка в CustomMeanShiftTracker: {e}")
            return False, None, 0


def create_test_videos():
    """Создает набор тестовых видео для анализа"""
    videos_info = []

    # 1. Простое движение (базовый тест)
    video1 = "test_simple_motion.mp4"
    if not os.path.exists(video1):
        print("Создаем видео с простым движением...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video1, fourcc, 25.0, (800, 600))

        for i in range(100):
            frame = np.full((600, 800, 3), 50, dtype=np.uint8)

            # Простая траектория
            center_x = 100 + int(600 * (i / 100))
            center_y = 300

            # Рисуем объект
            cv2.rectangle(frame, (center_x - 40, center_y - 30), (center_x + 40, center_y + 30), (0, 0, 200), -1)
            cv2.rectangle(frame, (center_x - 40, center_y - 30), (center_x + 40, center_y + 30), (0, 0, 255), 2)

            out.write(frame)
        out.release()
        videos_info.append((video1, "Простое линейное движение", 25, 100))

    # 2. Движение с изменением масштаба
    video2 = "test_scale_change.mp4"
    if not os.path.exists(video2):
        print("Создаем видео с изменением масштаба...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video2, fourcc, 25.0, (800, 600))

        for i in range(100):
            frame = np.full((600, 800, 3), 50, dtype=np.uint8)

            center_x = 400
            center_y = 300
            size = 20 + int(60 * abs(np.sin(i * 0.1)))

            cv2.rectangle(frame, (center_x - size, center_y - size), (center_x + size, center_y + size), (0, 200, 0),
                          -1)
            cv2.rectangle(frame, (center_x - size, center_y - size), (center_x + size, center_y + size), (0, 255, 0), 2)

            out.write(frame)
        out.release()
        videos_info.append((video2, "Изменение масштаба", 25, 100))

    # 3. Быстрое движение
    video3 = "test_fast_motion.mp4"
    if not os.path.exists(video3):
        print("Создаем видео с быстрым движением...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video3, fourcc, 25.0, (800, 600))

        for i in range(100):
            frame = np.full((600, 800, 3), 50, dtype=np.uint8)

            center_x = 100 + (i * 7) % 600
            center_y = 100 + int(100 * np.sin(i * 0.3))

            cv2.rectangle(frame, (center_x - 30, center_y - 30), (center_x + 30, center_y + 30), (200, 0, 0), -1)
            cv2.rectangle(frame, (center_x - 30, center_y - 30), (center_x + 30, center_y + 30), (255, 0, 0), 2)

            out.write(frame)
        out.release()
        videos_info.append((video3, "Быстрое движение", 25, 100))

    # 4. Движение с occlusion (перекрытием)
    video4 = "test_occlusion.mp4"
    if not os.path.exists(video4):
        print("Создаем видео с перекрытием...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video4, fourcc, 25.0, (800, 600))

        for i in range(150):
            frame = np.full((600, 800, 3), 50, dtype=np.uint8)

            # Основной объект
            center_x = 100 + int(400 * (i / 150))
            center_y = 300

            cv2.rectangle(frame, (center_x - 40, center_y - 30), (center_x + 40, center_y + 30), (0, 0, 200), -1)
            cv2.rectangle(frame, (center_x - 40, center_y - 30), (center_x + 40, center_y + 30), (0, 0, 255), 2)

            # Перекрывающий объект (на некоторых кадрах)
            if 50 <= i <= 100:
                occlusion_x = center_x + 20
                cv2.rectangle(frame, (occlusion_x - 50, center_y - 40), (occlusion_x + 50, center_y + 40),
                              (100, 100, 100), -1)

            out.write(frame)
        out.release()
        videos_info.append((video4, "Движение с перекрытием", 25, 150))

    # 5. Сложная траектория
    video5 = "test_complex_trajectory.mp4"
    if not os.path.exists(video5):
        print("Создаем видео со сложной траекторией...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video5, fourcc, 25.0, (800, 600))

        for i in range(120):
            frame = np.full((600, 800, 3), 50, dtype=np.uint8)

            t = i / 30
            center_x = 400 + int(200 * np.cos(t) * np.sin(2 * t))
            center_y = 300 + int(150 * np.sin(t) * np.cos(3 * t))

            cv2.rectangle(frame, (center_x - 35, center_y - 25), (center_x + 35, center_y + 25), (200, 200, 0), -1)
            cv2.rectangle(frame, (center_x - 35, center_y - 25), (center_x + 35, center_y + 25), (255, 255, 0), 2)

            out.write(frame)
        out.release()
        videos_info.append((video5, "Сложная траектория", 25, 120))

    return videos_info


def run_comparative_analysis():
    """Запуск сравнительного анализа трекеров"""
    print("\n" + "=" * 60)
    print("ЗАПУСК СРАВНИТЕЛЬНОГО АНАЛИЗА ТРЕКЕРОВ")
    print("=" * 60)

    # Создаем тестовые видео
    test_videos = create_test_videos()

    # Трекеры для сравнения
    trackers_to_test = ['CSRT', 'KCF', 'MOSSE']

    results = {}

    for video_path, description, fps, total_frames in test_videos:
        print(f"\nАнализ видео: {description}")
        print(f"   Файл: {video_path}")
        print(f"   Параметры: {fps} FPS, {total_frames} кадров")

        video_results = {}

        for tracker_type in trackers_to_test:
            print(f"\n   Тестирование трекера: {tracker_type}")

            # Запускаем трекинг
            metrics = run_single_tracking_test(video_path, tracker_type, description)
            video_results[tracker_type] = metrics

            print(f"      Успешных кадров: {metrics['success_rate']:.1f}%")
            print(f"      Среднее время обработки: {metrics['average_processing_time'] * 1000:.1f}ms")
            print(f"      Частота потерь: {metrics['tracking_loss_frequency']:.1f}%")

        results[description] = video_results

    # Сохраняем результаты
    save_comparison_results(results, test_videos)

    # Создаем сводную таблицу
    create_summary_table(results)

    return results


def run_single_tracking_test(video_path, tracker_type, video_description):
    """Запуск одного теста трекинга"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return None

    # Создаем систему трекинга
    tracking_system = TrackingSystem()

    if not tracking_system.initialize_tracker(tracker_type):
        cap.release()
        return None

    # Читаем первый кадр и выбираем объект автоматически
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    # Автоматический выбор bbox (центр кадра)
    height, width = frame.shape[:2]
    bbox = (width // 2 - 50, height // 2 - 50, 100, 100)

    if not tracking_system.init_tracking(frame, bbox):
        cap.release()
        return None

    # Обрабатываем все кадры
    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracking_system.update_tracking(frame)
        frame_count += 1

    cap.release()

    # Получаем метрики
    metrics = tracking_system.analyzer.calculate_metrics()
    return metrics


def run_custom_tracker_test():
    """Тестирование собственного трекера"""
    print("\nТЕСТИРОВАНИЕ СОБСТВЕННОЙ РЕАЛИЗАЦИИ TRACKER")

    test_videos = create_test_videos()
    custom_tracker_results = {}

    for video_path, description, fps, total_frames in test_videos:
        print(f"\nТестирование на видео: {description}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            continue

        # Создаем кастомный трекер
        custom_tracker = CustomMeanShiftTracker()
        analyzer = TrackingAnalyzer()

        # Инициализация
        ret, frame = cap.read()
        if not ret:
            cap.release()
            continue

        height, width = frame.shape[:2]
        bbox = (width // 2 - 50, height // 2 - 50, 100, 100)

        if not custom_tracker.init(frame, bbox):
            cap.release()
            continue

        # Обработка кадров
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            success, bbox, processing_time = custom_tracker.update(frame)
            if success:
                analyzer.update(True, bbox, processing_time)
            else:
                analyzer.update(False, None, processing_time)

            frame_count += 1

        cap.release()

        metrics = analyzer.calculate_metrics()
        custom_tracker_results[description] = metrics

        print(f"   Успешных кадров: {metrics['success_rate']:.1f}%")
        print(f"   Среднее время обработки: {metrics['average_processing_time'] * 1000:.1f}ms")

    return custom_tracker_results


def save_comparison_results(results, videos_info):
    """Сохранение результатов сравнения"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tracking_comparison_{timestamp}.json"

    output_data = {
        'test_date': timestamp,
        'test_videos': [
            {
                'description': desc,
                'file_path': path,
                'fps': fps,
                'total_frames': frames
            } for (path, desc, fps, frames) in videos_info
        ],
        'results': results
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены в: {filename}")
    return filename


def create_summary_table(results):
    """Создание сводной таблицы результатов"""
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ СРАВНИТЕЛЬНОГО АНАЛИЗА")
    print("=" * 80)

    # Заголовок таблицы
    print(
        f"\n{'Видео тест':<25} {'Трекер':<8} {'Успешность (%)':<15} {'Потери (%)':<12} {'Восстановление (%)':<18} {'Время (ms)':<12} {'Стабильность':<12}")
    print("-" * 110)

    for video_name, video_results in results.items():
        print(f"{video_name:<25}")
        for tracker_name, metrics in video_results.items():
            print(f"{'':<25} {tracker_name:<8} {metrics['success_rate']:<15.1f} "
                  f"{metrics['tracking_loss_frequency']:<12.1f} {metrics['recovery_rate']:<18.1f} "
                  f"{metrics['average_processing_time'] * 1000:<12.1f} {metrics['stability_score']:<12.1f}")


def run_interactive_demo():
    """Интерактивная демонстрация работы трекеров"""
    print("\nЗАПУСК ИНТЕРАКТИВНОЙ ДЕМОНСТРАЦИИ")

    # Создаем тестовые видео
    test_videos = create_test_videos()

    if not test_videos:
        print("Не удалось создать тестовые видео")
        return

    # Используем первое видео для демо
    video_path = test_videos[0][0]
    description = test_videos[0][1]

    print(f"Демонстрация на видео: {description}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    # Выбор трекера
    print("\nВыберите трекер для демонстрации:")
    print("   1. CSRT (точный)")
    print("   2. KCF (баланс скорости/точности)")
    print("   3. MOSSE (быстрый)")
    print("   4. Custom MeanShift (собственная реализация)")

    choice = input("Введите номер (1-4): ").strip()

    tracker_map = {'1': 'CSRT', '2': 'KCF', '3': 'MOSSE', '4': 'CUSTOM'}
    tracker_type = tracker_map.get(choice, 'KCF')

    if tracker_type == 'CUSTOM':
        # Используем кастомный трекер
        tracker = CustomMeanShiftTracker()
        use_custom = True
    else:
        # Используем библиотечный трекер
        tracking_system = TrackingSystem()
        tracking_system.initialize_tracker(tracker_type)
        use_custom = False

    # Инициализация
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    height, width = frame.shape[:2]
    bbox = (width // 2 - 50, height // 2 - 50, 100, 100)

    if use_custom:
        tracker.init(frame, bbox)
    else:
        tracking_system.init_tracking(frame, bbox)

    print(f"\nЗапуск демонстрации с трекером: {tracker_type}")
    print("Управление: q - выход, r - перезапуск, p - пауза")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()

            if use_custom:
                success, bbox, _ = tracker.update(frame)
            else:
                success, bbox = tracking_system.update_tracking(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if not use_custom:
                    # Рисуем траекторию для библиотечных трекеров
                    for i in range(1, len(tracking_system.trajectory)):
                        cv2.line(display_frame,
                                 tracking_system.trajectory[i - 1],
                                 tracking_system.trajectory[i],
                                 (255, 255, 0), 2)

            cv2.putText(display_frame, f"Tracker: {tracker_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Status: TRACKING" if success else "Status: LOST",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if success else (0, 0, 255), 2)

            cv2.imshow(f"Tracking Demo - {tracker_type}", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Перезапуск
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret:
                if use_custom:
                    tracker.init(frame, bbox)
                else:
                    tracking_system.init_tracking(frame, bbox)
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Главная функция"""
    print("СИСТЕМА АНАЛИЗА МЕТОДОВ ТРЕКИНГА")
    print("\nВыберите режим работы:")
    print("1. Полный сравнительный анализ (все трекеры + все видео)")
    print("2. Тестирование собственного трекера")
    print("3. Интерактивная демонстрация")
    print("4. Только создание тестовых видео")

    choice = input("Введите номер (1-4): ").strip()

    if choice == '1':
        # Полный сравнительный анализ
        results = run_comparative_analysis()

        # Дополнительно тестируем кастомный трекер
        print("\n" + "=" * 60)
        print("ДОПОЛНИТЕЛЬНОЕ ТЕСТИРОВАНИЕ СОБСТВЕННОГО ТРЕКЕРА")
        print("=" * 60)

        custom_results = run_custom_tracker_test()

        # Добавляем кастомный трекер в результаты
        for video_name, metrics in custom_results.items():
            if video_name in results:
                results[video_name]['CUSTOM'] = metrics

        # Обновляем таблицу
        create_summary_table(results)

    elif choice == '2':
        # Только тестирование кастомного трекера
        custom_results = run_custom_tracker_test()

        # Выводим результаты
        print("\nРЕЗУЛЬТАТЫ КАСТОМНОГО ТРЕКЕРА:")
        for video_name, metrics in custom_results.items():
            print(f"\n{video_name}:")
            print(f"  Успешность: {metrics['success_rate']:.1f}%")
            print(f"  Время обработки: {metrics['average_processing_time'] * 1000:.1f}ms")
            print(f"  Стабильность: {metrics['stability_score']:.1f}%")

    elif choice == '3':
        # Интерактивная демонстрация
        run_interactive_demo()

    elif choice == '4':
        # Только создание видео
        videos = create_test_videos()
        print(f"\nСоздано {len(videos)} тестовых видео:")
        for path, desc, fps, frames in videos:
            print(f"   {desc}: {path}")
    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()
    print("\nПРОГРАММА ЗАВЕРШЕНА!")