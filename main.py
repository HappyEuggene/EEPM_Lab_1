# Перейменуйте змінні
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

prices = np.array([1.25, 1.57, 1.81, 2.09, 2.45, 2.8, 3.19, 3.58, 3.85, 4.25, 4.62, 5])
demand_quantity = np.array([115, 109, 90, 85, 75, 58, 53, 40, 33, 30, 20, 14])
supply_quantity = np.array([17, 40, 62, 80, 100, 117, 131, 145, 156, 165, 170, 172])

def poly_fit(x, a, b, c):
    return a * x**2 + b * x + c

demand_params, _ = curve_fit(poly_fit, prices, demand_quantity)

supply_params, _ = curve_fit(poly_fit, prices, supply_quantity)

def equilibrium(p, subsidy=0):
    return poly_fit(p - subsidy, *demand_params) - poly_fit(p, *supply_params)

equilibrium_price = fsolve(equilibrium, 2.5)[0]  # Початкове наближення ціни

equilibrium_quantity = poly_fit(equilibrium_price, *demand_params)

subsidy = 1
new_equilibrium_price = fsolve(lambda p: equilibrium(p, subsidy), 2.5)[0]
new_equilibrium_quantity = poly_fit(new_equilibrium_price, *demand_params)

def arc_elasticity(q1, q2, p1, p2):
    dq = (q2 - q1) / ((q2 + q1) / 2)
    dp = (p2 - p1) / ((p2 + p1) / 2)
    return dq / dp

demand_elasticity = arc_elasticity(demand_quantity[-1], demand_quantity[0], prices[-1], prices[0])
supply_elasticity = arc_elasticity(supply_quantity[-1], supply_quantity[0], prices[-1], prices[0])

prices_plot = np.linspace(prices.min(), prices.max(), 100)
demand_plot = poly_fit(prices_plot, *demand_params)
supply_plot = poly_fit(prices_plot, *supply_params)
new_supply_plot = poly_fit(prices_plot - subsidy, *supply_params)  # Пропозиція з урахуванням дотації

plt.figure(figsize=(10, 6))
plt.plot(prices_plot, demand_plot, label='Попит', color='blue')
plt.plot(prices_plot, supply_plot, label='Пропозиція', color='red')
plt.plot(prices, demand_quantity, color='violet', linestyle='dashed', marker='o', label='Реальний попит')
plt.plot(prices, supply_quantity, color='black', linestyle='dashed', marker='o', label='Реальна пропозиція')
plt.plot(equilibrium_price, equilibrium_quantity, 'go', label='Точка рівноваги')
plt.title('Функції попиту та пропозиції')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(prices_plot, demand_plot, label='Попит', color='blue')
plt.plot(prices_plot, supply_plot, label='Пропозиція без дотації', color='red', linestyle='--')
plt.plot(prices_plot, new_supply_plot, label='Пропозиція з дотацією', color='green')
plt.scatter(equilibrium_price, equilibrium_quantity, color='black', zorder=5, label='Точка рівноваги без дотації')
plt.scatter(new_equilibrium_price, new_equilibrium_quantity, color='orange', zorder=5, label='Точка рівноваги з дотацією')
plt.title('Зміна ринкової рівноваги після введення дотації')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.show()

demand_slope = 2 * demand_params[0] * equilibrium_price + demand_params[1]
supply_slope = 2 * supply_params[0] * equilibrium_price + supply_params[1]
print("Нахил кривої попиту в точці рівноваги:", demand_slope)
print("Нахил кривої пропозиції в точці рівноваги:", supply_slope)
if supply_slope > demand_slope:
    print("Стан рівноваги стабільний.")
else:
    print("Стан рівноваги нестабільний.")

print(f"Функція попиту: Q_d = {demand_params[0]:.2f}P^2 + {demand_params[1]:.2f}P + {demand_params[2]:.2f}")
print(f"Функція пропозиції: Q_s = {supply_params[0]:.2f}P^2 + {supply_params[1]:.2f}P + {supply_params[2]:.2f}")

print(f"Точка рівноваги: Ціна = {equilibrium_price:.2f}, Кількість = {equilibrium_quantity:.2f}")
print(f"Дугова еластичність попиту: {demand_elasticity:.2f}")
print(f"Дугова еластичність пропозиції: {supply_elasticity:.2f}")
