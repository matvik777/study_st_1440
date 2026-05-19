import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Гравитационный параметр Земли, км^3 / с^2
MU_EARTH = 398600.4418


# Время моделирования
# Моделируем движение на интервале 2 часа
t_start = 0
t_end = 2 * 3600  # секунды
num_points = 1000

t_eval = np.linspace(t_start, t_end, num_points)


# Начальное состояние первого космического аппарата

state_1 = np.array([
    7010, 0, 0,
    0, 7.546, 0
])


# второго 

state_2 = np.array([
    7000, 10, 0,
    0, 0, 7.546
])


# Порог потенциально опасного сближения, км
danger_threshold = 5.0

def spacecraft_motion(t, state):
    """
    Описывает движение одного космического аппарата
    в центральном гравитационном поле Земли.

    """

    # Достаём координаты
    x = state[0]
    y = state[1]
    z = state[2]

    # Достаём скорости
    vx = state[3]
    vy = state[4]
    vz = state[5]

    # Расстояние от центра Земли до аппарата
    r = np.sqrt(x**2 + y**2 + z**2)

    # Гравитационное ускорение
    ax = -MU_EARTH * x / r**3
    ay = -MU_EARTH * y / r**3
    az = -MU_EARTH * z / r**3

    return [
        vx, vy, vz,
        ax, ay, az
    ]

# Численно рассчитываем траекторию первого аппарата
solution_1 = solve_ivp(
    fun=spacecraft_motion,
    t_span=(t_start, t_end),
    y0=state_1,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-9
)

# Численно рассчитываем траекторию второго аппарата
solution_2 = solve_ivp(
    fun=spacecraft_motion,
    t_span=(t_start, t_end),
    y0=state_2,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-9
)

# Проверяем, что интегрирование прошло успешно
if not solution_1.success:
    raise RuntimeError("Ошибка при расчёте траектории первого КА")

if not solution_2.success:
    raise RuntimeError("Ошибка при расчёте траектории второго КА")


# Достаём координаты аппаратов из решения
# solution.y имеет форму 6 x N:
# первые 3 строки — координаты x, y, z
# последние 3 строки — скорости vx, vy, vz
positions_1 = solution_1.y[0:3].T
positions_2 = solution_2.y[0:3].T


print("Траектория первого КА рассчитана:", positions_1.shape)
print("Траектория второго КА рассчитана:", positions_2.shape)

# Считаем относительный вектор между аппаратами
relative_positions = positions_1 - positions_2

# Считаем расстояние между аппаратами в каждый момент времени
distances = np.linalg.norm(relative_positions, axis=1)

print("Расстояния рассчитаны:", distances.shape)
print("Первое расстояние:", distances[0], "км")

# Находим индекс минимального расстояния
min_index = np.argmin(distances)

# Минимальное расстояние между аппаратами
min_distance = distances[min_index]

# Момент времени, когда произошло максимальное сближение
time_of_closest_approach = t_eval[min_index]

print("Минимальное расстояние:", min_distance, "км")
print("Момент максимального сближения:", time_of_closest_approach, "с")

# Классифицируем событие сближения
if min_distance < danger_threshold:
    status = "ПОТЕНЦИАЛЬНО ОПАСНОЕ СБЛИЖЕНИЕ"
else:
    status = "Безопасное сближение в рамках модели"


print("Порог опасности:", danger_threshold, "км")
print("Статус:", status)

# Создаём папку для результатов
os.makedirs("results", exist_ok=True)

# Сохраняем траектории и расстояние между КА
trajectories_data = np.column_stack([
    t_eval,
    positions_1,
    positions_2,
    distances
])

np.savetxt(
    "results/trajectories.csv",
    trajectories_data,
    delimiter=",",
    header="time_s,x1_km,y1_km,z1_km,x2_km,y2_km,z2_km,distance_km",
    comments="",
    fmt="%.6f"
)

# Сохраняем краткий итог расчёта
summary_data = np.array([
    [min_distance, time_of_closest_approach, danger_threshold]
])

np.savetxt(
    "results/summary.csv",
    summary_data,
    delimiter=",",
    header="min_distance_km,time_of_closest_approach_s,danger_threshold_km",
    comments="",
    fmt="%.6f"
)

print("Результаты сохранены в папку results")