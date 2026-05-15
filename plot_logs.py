import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ant_performance_log.csv")

plt.figure(figsize=(10, 5))
plt.plot(df["Step"], df["Reward"], label="Reward", color='blue')
plt.plot(df["Step"], df["TorsoHeight"], label="Torso Height", color='green')
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.title("Evaluation Performance of Ant")
plt.tight_layout()
plt.savefig("evaluation_plot.png")
plt.show()
