from radar_altimeter import RadarAltimeter, SurfaceParameters, SurfaceType, SurfaceGradient, PolarizationType
import matplotlib.pyplot as plt
import numpy as np
import torch

def test_simulation():
    # Создание экземпляра радиовысотомера
    altimeter = RadarAltimeter(use_gpu=torch.cuda.is_available())
    
    # Тестовые параметры
    heights = [500, 1000, 2000]  # метры
    roll_angles = [0, 15, 30]    # градусы
    pitch_angles = [-5, 0, 5]    # градусы
    polarizations = [PolarizationType.VERTICAL, PolarizationType.HORIZONTAL]
    
    # Тестовые типы поверхностей
    surfaces = [
        ("Сухой лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.8, moisture=0.2, temperature=25)),
        ("Влажный лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.8, moisture=0.8, temperature=20)),
        ("Лес после дождя", SurfaceParameters(SurfaceType.FOREST, roughness=0.9, moisture=0.95, temperature=18)),
        ("Зимний лес", SurfaceParameters(SurfaceType.FOREST, roughness=0.7, moisture=0.3, temperature=-10))
    ]
    
    # Создание градиентов поверхности
    height_map = np.zeros((100, 100))
    gradient = SurfaceGradient(height_map)
    
    # Тестирование для разных высот и поляризаций
    for height in heights:
        for polarization in polarizations:
            # Создание графика
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f'Тестирование модели отражения сигнала\nВысота: {height}м, Поляризация: {polarization.value}', fontsize=16)
            
            # Тестирование для разных типов поверхности
            for i, (name, params) in enumerate(surfaces):
                # Расчет угла скольжения
                base_grazing_angle = np.arctan(1.0 / height)
                roll_rad = np.radians(15)  # Фиксированный крен для теста
                pitch_rad = np.radians(0)  # Фиксированный тангаж для теста
                grazing_angle = np.arccos(np.cos(base_grazing_angle) * np.cos(roll_rad) * np.cos(pitch_rad))
                
                # Моделирование отражения с учетом многолучевого распространения
                t, rx_signal = altimeter.simulate_multipath_reflection(height, params, gradient, max_reflections=2)
                _, tx_signal = altimeter.generate_chirp_signal(altimeter.T, altimeter.B)
                
                # Расчет коэффициента отражения
                reflection_coeff = altimeter.calculate_reflection_coefficient(grazing_angle, params, polarization)
                
                # Построение графиков
                ax = axes[i//2, i%2]
                ax.plot(t * 1e6, tx_signal, 'r--', label='Передаваемый', alpha=0.5)
                ax.plot(t * 1e6, rx_signal, 'b-', label='Отраженный')
                ax.set_title(f'{name}\nКоэф. отражения: {reflection_coeff:.3f}')
                ax.set_xlabel('Время (мкс)')
                ax.set_ylabel('Амплитуда')
                ax.grid(True)
                ax.legend()
            
            plt.tight_layout()
            plt.show()
    
    # Тестирование зависимости от угла скольжения
    plt.figure(figsize=(10, 6))
    grazing_angles = np.linspace(0, np.pi/2, 100)
    
    for name, params in surfaces:
        for polarization in polarizations:
            reflection_coeffs = []
            for angle in grazing_angles:
                coeff = altimeter.calculate_reflection_coefficient(angle, params, polarization)
                reflection_coeffs.append(coeff)
            
            plt.plot(np.degrees(grazing_angles), reflection_coeffs, 
                    label=f'{name} ({polarization.value})')
    
    plt.title('Зависимость коэффициента отражения от угла скольжения')
    plt.xlabel('Угол скольжения (градусы)')
    plt.ylabel('Коэффициент отражения')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Тестирование многолучевого распространения
    plt.figure(figsize=(12, 8))
    max_reflections = [1, 2, 3, 4, 5]
    
    for name, params in surfaces:
        for reflections in max_reflections:
            t, rx_signal = altimeter.simulate_multipath_reflection(1000, params, gradient, reflections)
            plt.plot(t * 1e6, rx_signal, label=f'{name} ({reflections} отражений)')
    
    plt.title('Влияние количества отражений на сигнал')
    plt.xlabel('Время (мкс)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_simulation() 