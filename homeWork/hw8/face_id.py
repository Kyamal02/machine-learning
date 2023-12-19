import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime

# Путь к папке с изображениями лиц
path = 'D:\machine-learning\homeWork\hw8\KnowFaces'
images = []  # Список для хранения загруженных изображений
classNames = []  # Список для хранения имен классов (людей)
myList = os.listdir(path)  # Получение списка файлов в указанной директории
print(myList)  # Вывод списка файлов

# Цикл для чтения каждого изображения в директории
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')  # Чтение изображения
    images.append(curImg)  # Добавление изображения в список
    classNames.append(os.path.splitext(cls)[0])  # Добавление имени файла (без расширения) в список имен

print(classNames)  # Вывод списка имен

# Функция для нахождения кодировок лиц в изображениях
def findEncodings(images):
    encodeList = []  # Список для хранения кодировок
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертация изображения из BGR в RGB
        encode = face_recognition.face_encodings(img)[0]  # Получение кодировки лица
        encodeList.append(encode)  # Добавление кодировки в список
    return encodeList  # Возвращение списка кодировок

# Функция для отметки посещаемости
def markAttendance(name):
    with open("Attendance.csv", "r+") as f:  # Открытие файла для чтения и записи
        myDataList = f.readlines()  # Чтение всех строк файла
        nameList = []  # Список для хранения имен
        for line in myDataList:
            entry = line.split(',')  # Разбиение строки по запятой
            nameList.append(entry[0])  # Добавление имени в список
        if name not in nameList:  # Проверка, есть ли имя в списке
            now = datetime.now()  # Получение текущего времени
            dtString = now.strftime("%H:%M:%S")  # Форматирование времени
            f.writelines(f'\n{name}, {dtString}')  # Запись имени и времени в файл

# Получение кодировок известных лиц
encodeListKnown = findEncodings(images)
print("Декодирование закончено")  # Вывод сообщения о завершении декодирования

# Начало захвата видео с веб-камеры
cap = cv2.VideoCapture(0)

frame_skip = 5  # Указание обрабатывать каждый 5-й кадр
frame_counter = 0  # Инициализация счетчика кадров


while True:
    success, img = cap.read()  # Чтение кадра из видеопотока
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Изменение размера изображения для ускорения обработки
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Конвертация изображения из BGR в RGB

    facesCurFrame = face_recognition.face_locations(imgS)  # Определение местоположений лиц в текущем кадре
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # Получение кодировок лиц в текущем кадре

    # Цикл для сравнения каждого обнаруженного лица с известными кодировками
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.50)  # Сравнение лица с известными кодировками
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Расчет расстояния до известных кодировок
        matchIndex = np.argmin(faceDis)  # Получение индекса наименьшего расстояния

        y1, x2, y2, x1 = faceLoc  # Распаковка координат лица
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Корректировка координат с учетом измененного размера изображения

        if matches[matchIndex] and faceDis[matchIndex] < 0.50:  # расстояния
            name = classNames[matchIndex]  # Получение имени соответствующего лица
            markAttendance(name)  # Отметка посещаемости
        else:
            name = "Неизвестно"  # Обозначение неизвестного лица

        # Отображение имени и рамки вокруг лица
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("WebCam", img)  # Вывод обработанного изображения в окно
    cv2.waitKey(1)  # Ожидание нажатия клавиши для закрытия окна
