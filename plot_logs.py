import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ant_performance_log.csv")
plt.plot(df["Step"], df["Reward"], label = "Reward")
plt.plot(df["Step"], df["TorsoHeight"], label="Torso Height")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.title("Ant Performance Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()