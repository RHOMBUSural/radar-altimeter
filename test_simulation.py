from radar_altimeter import RadarAltimeter, SurfaceParameters, SurfaceType
import matplotlib.pyplot as plt
import numpy as np

def test_simulation():
    # Создание экземпляра радиовысотомера
    altimeter = RadarAltimeter()
    
    # Тестовые параметры
    heights = [500, 1000, 2000]  # метры
    roll_angles = [0, 15, 30]    # градусы
    pitch_angles = [-5, 0, 5]    # градусы
    
    # Тестовые типы леса
    forest_types = [
        ("Сухой лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.8, moisture=0.2, temperature=25)),
        ("Влажный лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.8, moisture=0.8, temperature=20)),
        ("Лес после дождя", SurfaceParameters(SurfaceType.FOREST, roughness=0.9, moisture=0.95, temperature=18)),
        ("Зимний лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.7, moisture=0.3, temperature=-10))
    ]
    
    # Создание графика
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Тестирование модели отражения сигнала', fontsize=16)
    
    # Тестирование для разных высот
    for i, height in enumerate(heights):
        for j, (name, params) in enumerate(forest_types):
            # Расчет угла скольжения
            base_grazing_angle = np.arctan(1.0 / height)
            roll_rad = np.radians(15)  # Фиксированный крен для теста
            pitch_rad = np.radians(0)  # Фиксированный тангаж для теста
            grazing_angle = np.arccos(np.cos(base_grazing_angle) * np.cos(roll_rad) * np.cos(pitch_rad))
            
            # Моделирование отражения
            t, rx_signal = altimeter.simulate_reflection(height, params)
            _, tx_signal = altimeter.generate_chirp_signal(altimeter.T, altimeter.B)
            
            # Расчет коэффициента отражения
            reflection_coeff = altimeter.calculate_reflection_coefficient(grazing_angle, params)
            
            # Построение графиков
            ax = axes[i, j]
            ax.plot(t * 1e6, tx_signal, 'r--', label='Передаваемый', alpha=0.5)
            ax.plot(t * 1e6, rx_signal, 'b-', label='Отраженный')
            ax.set_title(f'{name}\nВысота: {height}м, Крен: 15°\nКоэф. отражения: {reflection_coeff:.3f}')
            ax.set_xlabel('Время (мкс)')
            ax.set_ylabel('Амплитуда')
            ax.grid(True)
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Тестирование зависимости от угла скольжения
    plt.figure(figsize=(10, 6))
    grazing_angles = np.linspace(0, np.pi/2, 100)
    
    for name, params in forest_types:
        reflection_coeffs = []
        for angle in grazing_angles:
            coeff = altimeter.calculate_reflection_coefficient(angle, params)
            reflection_coeffs.append(coeff)
        
        plt.plot(np.degrees(grazing_angles), reflection_coeffs, label=name)
    
    plt.title('Зависимость коэффициента отражения от угла скольжения')
    plt.xlabel('Угол скольжения (градусы)')
    plt.ylabel('Коэффициент отражения')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_simulation() 