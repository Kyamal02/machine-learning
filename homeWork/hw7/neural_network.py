import time

import cv2
import keras
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


# Функция создает модель сверточной нейронной сети (CNN) для распознавания рукописных цифр.
def mnist_make_model(image_w: int, image_h: int):
    # image_w и image_h - ширина и высота изображений, соответственно.

    # Количество каналов изображения (1 для градаций серого)
    num_channels = 1
    # Количество классов (цифр от 0 до 9)
    num_classes = 10
    # Создается последовательная модель (Sequential), которая позволяет добавлять слои последовательно.
    model = Sequential()
    # Добавление сверточных слоев. Conv2D - это слой сверточной сети, который обычно используется для обработки изображений
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=(image_w, image_h, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Повторяем добавление сверточных слоев
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Преобразование двумерного массива признаков в одномерный
    model.add(Flatten())

    # Добавление полносвязных слоев для классификации. Dense - это тип слоя, который полностью связан с предыдущим слоем
    model.add(Dense(128, activation='relu'))
    # Выходной слой с 10 нейронами (по одному на каждый класс цифр)
    model.add(Dense(num_classes, activation='softmax'))

    # Компиляция модели. Adam - это метод оптимизации, а 'categorical_crossentropy' используется для многоклассовой классификации
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Функция для обучения модели на наборе данных MNIST
def mnist_mlp_train(model):
    # Загрузка набора данных MNIST
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

    # Размер изображений и количество каналов
    image_size = 28
    num_channels = 1

    # Предобработка и нормализация обучающих данных
    # Изменение формы данных изображений для соответствия ожидаемой форме входных данных модели (количество образцов, высота, ширина, количество каналов).
    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
    # Нормализация данных путем преобразования значений пикселей из диапазона 0-255 в диапазон 0-1, что облегчает обучение модели.
    train_data = train_data.astype('float32') / 255.0

    # Кодирование меток классов (цифр) в формат one-hot
    # One-hot кодирование преобразует категориальные метки в вектор, где каждая цифра представлена уникальным бинарным вектором.
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

    # Предобработка тестовых данных
    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

    print("Training the network...")
    t_start = time.time()

    # Обучение модели
    model.fit(train_data, train_labels_cat, epochs=8, batch_size=64, validation_data=(val_data, val_labels_cat))

    print("Done, dT:", time.time() - t_start)

    return model


# Функция для выделения отдельных цифр из изображения
def segment_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_contours = []
    minus_signs = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w / h > 1.5:  # Предполагается, что это минус
            minus_signs.append(('minus', x))
        else:
            digit = thresh[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (28, 28))
            digit_contours.append((x, resized_digit))

    # Сортируем только изображения цифр
    digit_contours.sort(key=lambda item: item[0])

    # Объединяем знаки минуса и отсортированные цифры
    all_items = minus_signs + digit_contours
    return all_items



def display_digits(image_path):
    digit_images = segment_image(image_path)

    for i, item in enumerate(digit_images):
        if item[0] == 'minus':  # Пропускаем знак минуса
            continue
        digit = item[1]  # Получаем изображение цифры
        cv2.imshow(f'Digit {i}', digit)
        cv2.waitKey(0)

    cv2.destroyAllWindows()



def mlp_digits_predict(model, digit_images):
    # Функция для предсказания цифр с помощью модели
    result_arr = []
    for item in digit_images:
        if item[0] == 'minus':  # Проверка, является ли элемент знаком минуса
            result_arr.append('-')
        else:
            digit = item[1]
            img_arr = np.expand_dims(digit, axis=0)
            img_arr = img_arr.reshape((1, 28, 28, 1))
            result = model.predict([img_arr])
            predicted_class = np.argmax(result, axis=1)
            result_arr.append(str(predicted_class[0]))

    # Формирование строки с результатами и ее вывод
    result_str = ''.join(result_arr)  # Используем '' для соединения без пробелов
    print(result_str)


if __name__ == "__main__":
    # model = mnist_make_model(image_w=28, image_h=28)
    # mnist_mlp_train(model)
    # model.save('cnn_digits_28x28.h5')
    digit_images = segment_image("saved_drawing.png")
    model = keras.models.load_model('cnn_digits_28x28.h5')
    mlp_digits_predict(model, digit_images)
    display_digits('saved_drawing.png')
