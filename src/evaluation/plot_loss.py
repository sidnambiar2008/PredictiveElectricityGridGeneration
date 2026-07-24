import matplotlib.pyplot as plt
import os

# 1. Enter the 15 numbers from your training log screenshot here:
training_loss = [
    0.014029, 0.005762, 0.004442, 0.003737, 0.003268,
    0.002940, 0.002667, 0.002405, 0.002154, 0.001912,
    0.001736, 0.001640, 0.001568, 0.001507, 0.001459
]

epochs = list(range(1, 16))

# 2. Build a clean, professional visualization canvas
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss, marker='o', color='red', linewidth=2.5, markersize=6, label="MSE Training Loss")

# 3. Add styling elements to match professional presentation standards
plt.title("GridPulse LSTM Model Convergence Profile: 15-Epoch Optimization Pass for PJM", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Training Epoch Milestone", fontsize=11, labelpad=8)
plt.ylabel("Mean Squared Error (Loss Score)", fontsize=11, labelpad=8)
plt.xticks(epochs)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()

# 4. Save a high-res PNG file directly to your visuals folder for your report
os.makedirs("visuals", exist_ok=True)
output_path = "visuals/lstm_training_loss_curve.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f" Success! The loss curve graphic has been saved to: {output_path}")
