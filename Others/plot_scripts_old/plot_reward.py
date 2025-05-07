# import numpy as np
# import matplotlib.pyplot as plt

# # Define error values
# error = np.linspace(0, 2, 100)  # Error values from 0 to 2

# # Define scale values to compare different curves
# scale_values = [1, 2, 5]

# # Create a figure
# plt.figure(figsize=(8, 5))

# # Plot exp(-scale * error) and 1/(1 + scale * error) for different scales
# for scale in scale_values:
#     reward_exp = np.exp(-scale * error)  # Exponential function
#     reward_inverse = 1 / (1 + scale * error)  # Inverse function
    
#     plt.plot(error, reward_exp, linestyle='-', label=f'$e^{{-{scale} \cdot error}}$')
#     plt.plot(error, reward_inverse, linestyle='--', label=f'$1 / (1 + {scale} \cdot error)$')

# # Formatting the plot
# plt.xlabel('Error')
# plt.ylabel('Reward')
# plt.title(r'Comparison of $e^{-\text{scale} \cdot \text{error}}$ and $\frac{1}{1 + \text{scale} \cdot \text{error}}$')
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Define a range for success_rate
success_rate = np.linspace(0, 1, 500)  # Adjust range and number of points as needed

# Calculate the function values: np.exp(-0.001*success_rate)+0.5
y = np.exp(-10.0 * success_rate) + 0.5

# Create the plot
plt.plot(success_rate, y)
plt.xlabel('Success Rate')
plt.ylabel('Function Value')
plt.title('Plot of np.exp(-10.0 * success_rate) + 0.5')
plt.grid(True)
plt.show()
