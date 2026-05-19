import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Загружаем подробные данные по траекториям
trajectories = np.loadtxt(
    "results/trajectories.csv",
    delimiter=",",
    skiprows=1
)

# Загружаем краткие итоговые данные
summary = np.loadtxt(
    "results/summary.csv",
    delimiter=",",
    skiprows=1
)

# Если summary считался как одномерный массив, это нормально
min_distance = summary[0]
time_of_closest_approach = summary[1]
danger_threshold = summary[2]

# Распаковываем колонки trajectories.csv
time_s = trajectories[:, 0]

x1 = trajectories[:, 1]
y1 = trajectories[:, 2]
z1 = trajectories[:, 3]

x2 = trajectories[:, 4]
y2 = trajectories[:, 5]
z2 = trajectories[:, 6]

distance_km = trajectories[:, 7]

print("Данные успешно загружены")
print("Количество точек:", len(time_s))
print("Минимальное расстояние:", min_distance, "км")
print("Момент сближения:", time_of_closest_approach / 60, "мин")

# Переводим время из секунд в минуты для удобства графика
time_min = time_s / 60
time_ca_min = time_of_closest_approach / 60


# Строим график расстояния между аппаратами
plt.figure(figsize=(10, 5))

plt.plot(
    time_min,
    distance_km,
    label="Расстояние между КА"
)

# Отмечаем точку максимального сближения
plt.scatter(
    time_ca_min,
    min_distance,
    zorder=5,
    label="Минимальное расстояние"
)

# Показываем порог опасности
plt.axhline(
    danger_threshold,
    linestyle="--",
    label="Порог опасности"
)

plt.xlabel("Время, мин")
plt.ylabel("Расстояние, км")
plt.title("Расстояние между двумя космическими аппаратами")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("results/distance_vs_time.png", dpi=300)
plt.show()
# Строим 3D-график траекторий двух аппаратов
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

# Траектории
ax.plot(x1, y1, z1, label="Траектория КА-1")
ax.plot(x2, y2, z2, label="Траектория КА-2")

# Начальные точки
ax.scatter(x1[0], y1[0], z1[0], label="Старт КА-1")
ax.scatter(x2[0], y2[0], z2[0], label="Старт КА-2")

# Положение аппаратов в момент максимального сближения
closest_index = np.argmin(distance_km)

ax.scatter(
    x1[closest_index], y1[closest_index], z1[closest_index],
    label="КА-1 в момент сближения"
)

ax.scatter(
    x2[closest_index], y2[closest_index], z2[closest_index],
    label="КА-2 в момент сближения"
)

# Рисуем Землю как сферу
earth_radius = 6371  # км

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)

earth_x = earth_radius * np.outer(np.cos(u), np.sin(v))
earth_y = earth_radius * np.outer(np.sin(u), np.sin(v))
earth_z = earth_radius * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.4)

# Одинаковый масштаб по всем осям
all_x = np.concatenate([x1, x2])
all_y = np.concatenate([y1, y2])
all_z = np.concatenate([z1, z2])

max_range = max(
    np.max(np.abs(all_x)),
    np.max(np.abs(all_y)),
    np.max(np.abs(all_z))
)

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("X, км")
ax.set_ylabel("Y, км")
ax.set_zlabel("Z, км")
ax.set_title("Траектории двух космических аппаратов")
ax.legend()

plt.tight_layout()
plt.savefig("results/trajectories_3d.png", dpi=300)
plt.show()