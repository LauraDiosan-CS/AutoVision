import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(r"exp/det_bdd_v11_custom2/results.csv")
df[["train/box_loss","train/cls_loss","train/dfl_loss"]].plot()
plt.title("Training Losses"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()