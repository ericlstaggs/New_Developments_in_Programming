import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#Task: 1 Symbolid derivative
x_sym = sp.symbols('x')

#functions
f1_sym = sp.sin(x_sym)
f2_sym = sp.log(x_sym)

#get derivatives
dif1 = sp.diff(f1_sym,x_sym)
dif2 = sp.diff(f2_sym,x_sym)

print(f"1. function: sin(x) -> Derivative: {f1_sym}")

print(f"2. function: log(x) -> Derivative: {f2_sym}")

#Task 2: Plot derivatives

x_vals = np.linspace(.1, 2 * np.pi, 500)

#Y
y_sin = np.sin(x_vals)
dy_sin = np.cos(x_vals)

y_log = np.log(x_vals)
dy_log = 1 / x_vals

#plot
plt.plot(x_vals, y_sin, "r--", label="f(x) = \sin(x)")
plt.plot(x_vals, dy_sin, "p--", label="f(x) = \cos(x)")

plt.plot(x_vals, dy_log, "b--", label="(x) = \ln(x)")
plt.plot(x_vals, y_log, "g--", label="(x) = \frac(x)")


plt.title("f(x) and derivatives")
plt.xlabel("x Axis")
plt.ylabel("y Axis")

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=11, shadow=True)
plt.show()
