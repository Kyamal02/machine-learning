import time

import cv2
import keras
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


def mnist_make_model(image_w: int, image_h: int):
    image_size = 28
    num_channels = 1  # 1 for grayscale images
    num_classes = 10  # Number of outputs
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     padding='same',
                     input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def mnist_mlp_train(model):
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

    # Get image size
    image_size = 28
    num_channels = 1  # 1 for grayscale images

    # re-shape and re-scale the images data
    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
    train_data = train_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

    # re-shape and re-scale the images validation data
    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

    print("Training the network...")
    t_start = time.time()

    # Start training the network
    model.fit(train_data, train_labels_cat, epochs=8, batch_size=64,
              validation_data=(val_data, val_labels_cat))

    print("Done, dT:", time.time() - t_start)

    return model


def segment_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Бинаризация изображения
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

    # Нахождение контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем список для хранения данных о контурах и их ограничивающих прямоугольниках
    digits_with_bounding_boxes = []

    for contour in contours:
        # Создание ограничивающего прямоугольника для каждого контура
        x, y, w, h = cv2.boundingRect(contour)
        # Выделение области с цифрой
        digit = thresh[y:y + h, x:x + w]
        # Масштабирование до размера 28x28
        resized_digit = cv2.resize(digit, (28, 28))
        digits_with_bounding_boxes.append((x, resized_digit))

    # Сортировка цифр по координате x
    digits_with_bounding_boxes.sort(key=lambda item: item[0])

    # Возвращаем только изображения цифр
    return [digit[1] for digit in digits_with_bounding_boxes]


def display_digits(image_path):
    digit_images = segment_image(image_path)

    for i, digit in enumerate(digit_images):
        cv2.imshow(f'Digit {i}', digit)
        cv2.waitKey(0)  # Ожидание нажатия клавиши перед показом следующей цифры

    cv2.destroyAllWindows()

def mlp_digits_predict(model, digit_images):
    result_arr = []
    for digit in digit_images:
        img_arr = np.expand_dims(digit, axis=0)
        img_arr = img_arr.reshape((1, 28, 28, 1))

        result = model.predict([img_arr])
        predicted_class = np.argmax(result, axis=1)
        result_arr.append(predicted_class)
    result_str = ' '.join(str(digit) for digit in result_arr)
    print(result_str)


if __name__ == "__main__":
    # model = mnist_make_model(image_w=28, image_h=28)
    # mnist_mlp_train(model)
    # model.save('cnn_digits_28x28.h5')
    digit_images = segment_image("saved_drawing.png")
    model = keras.models.load_model('cnn_digits_28x28.h5')
    mlp_digits_predict(model, digit_images)
    display_digits('saved_drawing.png')
