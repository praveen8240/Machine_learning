import matplotlib.pyplot as plt
import seaborn as sns

# Example data
data = [12, 7, 9, 15, 14, 9, 13, 16, 11, 7]

# Create a box plot
plt.figure(figsize=(5, 5))
sns.boxplot(data=data)
plt.title("Box Plot")
plt.show()
