import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(42)

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"Обучающая выборка: {train_images.shape}")
print(f"Тестовая выборка: {test_images.shape}")
print(f"Метки обучающей выборки: {train_labels.shape}")

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)


def plot_samples(images, labels, num_samples=10):
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {np.argmax(labels[i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


print("\nПримеры изображений из обучающей выборки:")
plot_samples(train_images, train_labels)

# Задание 1: Многослойный персептрон (MLP)

def create_mlp_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(10, activation='softmax')
    ])

    return model


mlp_model = create_mlp_model()

mlp_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "=" * 60)
print("МОДЕЛЬ МНОГОСЛОЙНОГО ПЕРСЕПТРОНА (MLP)")
print("=" * 60)
mlp_model.summary()

# Задание 2: Исследование влияния количества эпох

def train_and_evaluate_epochs(model, train_data, train_labels,
                              test_data, test_labels, epochs_list):

    results = []

    for epochs in epochs_list:
        print(f"\n{'=' * 60}")
        print(f"Обучение модели с {epochs} эпохами")
        print(f"{'=' * 60}")

        current_model = keras.models.clone_model(model)
        current_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        start_time = time.time()

        history = current_model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1,
            verbose=0
        )

        training_time = time.time() - start_time

        test_loss, test_accuracy = current_model.evaluate(
            test_data, test_labels, verbose=0
        )

        start_pred_time = time.time()
        predictions = current_model.predict(test_data[:1000], verbose=0)
        prediction_time = (time.time() - start_pred_time) / 1000

        results.append({
            'epochs': epochs,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'history': history.history
        })

        print(f"Время обучения: {training_time:.2f} сек")
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        print(f"Время предсказания на один образец: {prediction_time * 1000:.4f} мс")

    return results


epochs_to_test = [5, 10, 15, 20, 25]

print("\n" + "=" * 60)
print("ИССЛЕДОВАНИЕ ВЛИЯНИЯ КОЛИЧЕСТВА ЭПОХ (MLP)")
print("=" * 60)

mlp_results = train_and_evaluate_epochs(
    mlp_model, train_images, train_labels, test_images, test_labels, epochs_to_test
)


def plot_epochs_comparison(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = [r['epochs'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]

    axes[0, 0].plot(epochs, accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Количество эпох')
    axes[0, 0].set_ylabel('Точность на тестовой выборке')
    axes[0, 0].set_title('Точность в зависимости от количества эпох')
    axes[0, 0].grid(True, alpha=0.3)

    training_times = [r['training_time'] for r in results]
    axes[0, 1].plot(epochs, training_times, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Количество эпох')
    axes[0, 1].set_ylabel('Время обучения (сек)')
    axes[0, 1].set_title('Время обучения в зависимости от количества эпох')
    axes[0, 1].grid(True, alpha=0.3)

    pred_times = [r['prediction_time'] * 1000 for r in results]  # в мс
    axes[1, 0].plot(epochs, pred_times, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Количество эпох')
    axes[1, 0].set_ylabel('Время предсказания (мс на образец)')
    axes[1, 0].set_title('Скорость предсказания')
    axes[1, 0].grid(True, alpha=0.3)

    efficiency = [acc / time for acc, time in zip(accuracies, training_times)]
    axes[1, 1].plot(epochs, efficiency, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Количество эпох')
    axes[1, 1].set_ylabel('Точность / время обучения')
    axes[1, 1].set_title('Эффективность обучения')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ДЛЯ РАЗНОГО КОЛИЧЕСТВА ЭПОХ")
    print("=" * 80)
    print(f"{'Эпохи':<10} {'Точность':<12} {'Время обуч.':<15} {'Время предск.':<15}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['epochs']:<10} {r['test_accuracy']:.4f}      {r['training_time']:<10.2f} сек    {r['prediction_time'] * 1000:<10.4f} мс")


plot_epochs_comparison(mlp_results)

# Задание 3: Сверточная нейронная сеть (CNN)


def create_cnn_model(architecture='standard'):
    model = models.Sequential()

    if architecture == 'simple':
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

    elif architecture == 'standard':
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

    elif architecture == 'deep':
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

    return model


train_images_cnn = train_images.reshape(-1, 28, 28, 1)
test_images_cnn = test_images.reshape(-1, 28, 28, 1)

cnn_architectures = ['simple', 'standard', 'deep']
cnn_results = []

print("\n" + "=" * 60)
print("СРАВНЕНИЕ РАЗНЫХ АРХИТЕКТУР CNN")
print("=" * 60)

for arch in cnn_architectures:
    print(f"\nАрхитектура: {arch.upper()}")
    print("-" * 40)

    cnn_model = create_cnn_model(architecture=arch)
    cnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cnn_model.summary()

    start_time = time.time()

    history = cnn_model.fit(
        train_images_cnn, train_labels,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )

    training_time = time.time() - start_time

    test_loss, test_accuracy = cnn_model.evaluate(
        test_images_cnn, test_labels, verbose=0
    )

    start_pred_time = time.time()
    predictions = cnn_model.predict(test_images_cnn[:1000], verbose=0)
    prediction_time = (time.time() - start_pred_time) / 1000

    cnn_results.append({
        'architecture': arch,
        'model': cnn_model,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'history': history.history,
        'num_params': cnn_model.count_params()
    })

    print(f"Количество параметров: {cnn_model.count_params():,}")
    print(f"Время обучения: {training_time:.2f} сек")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    print(f"Время предсказания на один образец: {prediction_time * 1000:.4f} мс")


print("\n" + "=" * 80)
print("ИТОГОВОЕ СРАВНЕНИЕ MLP И CNN")
print("=" * 80)

best_mlp_result = max(mlp_results, key=lambda x: x['test_accuracy'])

best_cnn_result = max(cnn_results, key=lambda x: x['test_accuracy'])

print(f"\nМНОГОСЛОЙНЫЙ ПЕРСЕПТРОН (MLP):")
print(f"  - Лучшая точность: {best_mlp_result['test_accuracy']:.4f}")
print(f"  - Время обучения: {best_mlp_result['training_time']:.2f} сек")
print(f"  - Оптимальное количество эпох: {best_mlp_result['epochs']}")
print(f"  - Время предсказания: {best_mlp_result['prediction_time'] * 1000:.4f} мс/образец")

print(f"\nСВЕРТОЧНАЯ СЕТЬ (CNN - {best_cnn_result['architecture']}):")
print(f"  - Лучшая точность: {best_cnn_result['test_accuracy']:.4f}")
print(f"  - Время обучения: {best_cnn_result['training_time']:.2f} сек")
print(f"  - Количество параметров: {best_cnn_result['num_params']:,}")
print(f"  - Время предсказания: {best_cnn_result['prediction_time'] * 1000:.4f} мс/образец")

print(f"\nУЛУЧШЕНИЕ ТОЧНОСТИ (CNN vs MLP): "
      f"{((best_cnn_result['test_accuracy'] - best_mlp_result['test_accuracy']) * 100):.2f}%")


def plot_architecture_comparison(mlp_results, cnn_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    architectures = ['MLP'] + [f'CNN\n({r["architecture"]})' for r in cnn_results]
    accuracies = [best_mlp_result['test_accuracy']] + [r['test_accuracy'] for r in cnn_results]

    bars = axes[0].bar(architectures, accuracies, color=['blue', 'green', 'orange', 'red'])
    axes[0].set_ylabel('Точность на тестовой выборке')
    axes[0].set_title('Сравнение точности разных архитектур')
    axes[0].set_ylim([0.9, 1.0])

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                     f'{acc:.4f}', ha='center', va='bottom')

    training_times = [best_mlp_result['training_time']] + [r['training_time'] for r in cnn_results]
    bars = axes[1].bar(architectures, training_times, color=['blue', 'green', 'orange', 'red'])
    axes[1].set_ylabel('Время обучения (сек)')
    axes[1].set_title('Сравнение времени обучения')

    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 2,
                     f'{time_val:.1f} с', ha='center', va='bottom')

    pred_times = [best_mlp_result['prediction_time'] * 1000] + [r['prediction_time'] * 1000 for r in cnn_results]
    bars = axes[2].bar(architectures, pred_times, color=['blue', 'green', 'orange', 'red'])
    axes[2].set_ylabel('Время предсказания (мс/образец)')
    axes[2].set_title('Сравнение скорости предсказания')

    for bar, time_val in zip(bars, pred_times):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                     f'{time_val:.4f} мс', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


plot_architecture_comparison([best_mlp_result], cnn_results)

# Демонстрация работы лучшей модели

best_model = best_cnn_result['model']

predictions = best_model.predict(test_images_cnn[:10])
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels[:10], axis=1)

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i], cmap='gray')

    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    plt.title(f"Прогноз: {predicted_labels[i]}\nИстина: {true_labels[i]}", color=color)
    plt.axis('off')
plt.suptitle(f'Примеры работы лучшей модели (CNN - {best_cnn_result["architecture"]})', fontsize=14)
plt.tight_layout()
plt.show()

all_predictions = best_model.predict(test_images_cnn)
all_predicted_labels = np.argmax(all_predictions, axis=1)
all_true_labels = np.argmax(test_labels, axis=1)

error_indices = np.where(all_predicted_labels != all_true_labels)[0]

print(f"\nАнализ ошибок лучшей модели:")
print(f"Всего ошибок: {len(error_indices)} из {len(test_images)}")
print(f"Точность: {(1 - len(error_indices) / len(test_images)):.4f}")

if len(error_indices) > 0:
    print(f"\nПримеры ошибочных предсказаний:")
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(error_indices[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[idx], cmap='gray')
        plt.title(f"Прогноз: {all_predicted_labels[idx]}\nИстина: {all_true_labels[idx]}", color='red')
        plt.axis('off')
    plt.suptitle('Примеры ошибочных предсказаний', fontsize=14)
    plt.tight_layout()
    plt.show()
