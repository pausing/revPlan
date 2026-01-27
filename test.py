import matplotlib.pyplot as plt
import numpy as np

# Example data
departments = ['HR', 'IT', 'Sales', 'Finance']
current_load = [90, 110, 70, 130]  # Current usage/load
max_capacity = [100, 100, 100, 100]  # Max capacity

# Split current_load into within and over capacity
within_capacity = [min(c, m) for c, m in zip(current_load, max_capacity)]
over_capacity = [max(0, c - m) for c, m in zip(current_load, max_capacity)]

# X positions
x = np.arange(len(departments))
bar_width = 0.5

# Plot base capacity bar (optional for reference)
plt.bar(x, max_capacity, width=bar_width, color='lightgrey', label='Max Capacity')

# Plot load within capacity
plt.bar(x, within_capacity, width=bar_width, color='blue', label='Current Load')

# Plot excess load (stacked on top)
plt.bar(x, over_capacity, width=bar_width, bottom=within_capacity, color='red', label='Overload')

# Labels and formatting
plt.xticks(x, departments)
plt.ylabel('Load')
plt.title('Department Load vs Max Capacity')
plt.legend()

plt.tight_layout()
plt.show()
