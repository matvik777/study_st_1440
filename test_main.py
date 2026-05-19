import numpy as np

from main import (
    MU_EARTH,
    spacecraft_motion,
    calculate_distances,
    find_closest_approach,
    classify_conjunction,
)


def test_spacecraft_motion_at_x_axis():
    """
    Проверяем движение КА в точке [7000, 0, 0].

    В этой точке гравитационное ускорение должно быть направлено
    только по оси X в отрицательную сторону.
    """

    state = np.array([
        7000, 0, 0,
        0, 7.5, 0
    ])

    result = spacecraft_motion(0, state)

    dx_dt = result[0]
    dy_dt = result[1]
    dz_dt = result[2]

    ax = result[3]
    ay = result[4]
    az = result[5]

    expected_ax = -MU_EARTH / 7000**2

    assert np.isclose(dx_dt, 0)
    assert np.isclose(dy_dt, 7.5)
    assert np.isclose(dz_dt, 0)

    assert np.isclose(ax, expected_ax)
    assert np.isclose(ay, 0)
    assert np.isclose(az, 0)


def test_calculate_distances_3_4_5():
    """
    Проверяем классический случай расстояния 3-4-5.
    """

    positions_1 = np.array([
        [0, 0, 0]
    ])

    positions_2 = np.array([
        [3, 4, 0]
    ])

    distances = calculate_distances(positions_1, positions_2)

    assert np.isclose(distances[0], 5)


def test_find_closest_approach():
    """
    Проверяем, что функция правильно находит минимум
    и соответствующий момент времени.
    """

    t_eval = np.array([0, 10, 20, 30])
    distances = np.array([100, 50, 5, 20])

    min_distance, time_of_closest_approach = find_closest_approach(
        t_eval,
        distances
    )

    assert min_distance == 5
    assert time_of_closest_approach == 20


def test_classify_conjunction_dangerous():
    """
    Если минимальное расстояние меньше порога,
    сближение должно быть опасным.
    """

    status = classify_conjunction(
        min_distance=2,
        danger_threshold=5
    )

    assert status == "ПОТЕНЦИАЛЬНО ОПАСНОЕ СБЛИЖЕНИЕ"


def test_classify_conjunction_safe():
    """
    Если минимальное расстояние больше порога,
    сближение должно быть безопасным.
    """

    status = classify_conjunction(
        min_distance=10,
        danger_threshold=5
    )

    assert status == "Безопасное сближение в рамках модели"