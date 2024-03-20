import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

price_array = np.array([1.25, 1.57, 1.81, 2.09, 2.45, 2.8, 3.19, 3.58, 3.85, 4.25, 4.62, 5])
demand_volume = np.array([115, 109, 90, 85, 75, 58, 53, 40, 33, 30, 20, 14])
supply_volume = np.array([17, 40, 62, 80, 100, 117, 131, 145, 156, 165, 170, 172])

def quadratic_fit(x, a, b, c):
    return a * x ** 2 + b * x + c

demand_coeffs = curve_fit(quadratic_fit, price_array, demand_volume)[0]
supply_coeffs = curve_fit(quadratic_fit, price_array, supply_volume)[0]

def find_balance(subvention=0):
    balance_func = lambda p: quadratic_fit(p - subvention, *demand_coeffs) - quadratic_fit(p, *supply_coeffs)
    equilibrium_cost = fsolve(balance_func, 2.5)[0]
    equilibrium_amount = quadratic_fit(equilibrium_cost, *demand_coeffs)
    return equilibrium_cost, equilibrium_amount

balance_price, balance_quantity = find_balance()
new_balance_price, new_balance_quantity = find_balance(subvention=1)

def arc_elasticity(q_start, q_end, p_start, p_end):
    delta_q = (q_end - q_start) / ((q_end + q_start) / 2)
    delta_p = (p_end - p_start) / ((p_end + p_start) / 2)
    return delta_q / delta_p

demand_elasticity = arc_elasticity(demand_volume[-1], demand_volume[0], price_array[-1], price_array[0])
supply_elasticity = arc_elasticity(supply_volume[-1], supply_volume[0], price_array[-1], price_array[0])

price_range = np.linspace(price_array.min(), price_array.max(), 100)

demand_curve = quadratic_fit(price_range, *demand_coeffs)
supply_curve = quadratic_fit(price_range, *supply_coeffs)
new_supply_curve = quadratic_fit(price_range - 1, *supply_coeffs)

plt.figure(figsize=(10, 6))
plt.plot(price_array, demand_volume, color='violet', linestyle='dashed', marker='o', label='Реальний попит')
plt.plot(price_array, supply_volume, color='black', linestyle='dashed', marker='o', label='Реальна пропозиція')
plt.plot(price_range, demand_curve, label='Попит', color='blue')
plt.plot(price_range, supply_curve, label='Пропозиція', color='red')
plt.scatter(balance_price, balance_quantity, color='green', zorder=5, label='Точка рівноваги без дотацій')
plt.title('Market Equilibrium')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 8))
plt.plot(price_range, demand_curve, label='Попит', color='blue')
plt.plot(price_range, supply_curve, linestyle='--', label='Пропозиція без дотацій', color='red')
plt.plot(price_range, new_supply_curve, label='Пропозиція з дотаціями', color='green')
plt.scatter(new_balance_price, new_balance_quantity, color='orange', zorder=5, label='Точка рівноваги з дотаціями')
plt.title('Impact of Subsidy on Market Equilibrium')
plt.xlabel('Ціна')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
demand_slope_at_equilibrium = 2 * demand_coeffs[0] * balance_price + demand_coeffs[1]
supply_slope_at_equilibrium = 2 * supply_coeffs[0] * balance_price + supply_coeffs[1]

equilibrium_stability = "Стабільна" if supply_slope_at_equilibrium > demand_slope_at_equilibrium else "Не стабільна"

# Print the results
print("Точка рівноваги:", equilibrium_stability)
print("Точка рівноваги: Ціна  =", balance_price, ", Кількість  =", balance_quantity)
print("Дугова еластичність попиту:", demand_elasticity)
print("Дугова еластичність пропозиції:", supply_elasticity)
print("Нахил кривої попиту в точці рівноваги:", demand_slope_at_equilibrium)
print("Нахил кривої пропозиції в точці рівноваги:", supply_slope_at_equilibrium)