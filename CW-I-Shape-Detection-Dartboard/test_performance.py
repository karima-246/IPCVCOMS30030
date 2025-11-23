import matplotlib.pyplot as plt
import numpy as np
import face

# Example data for 16 images
images = [f"Image {i}" for i in range(16)]
tpr_scores = face.tprs
f1_scores = face.f1s

# Calculate averages
avg_tpr = face.tpr_avg
avg_f1 = face.f1_avg

# Prepare table data
cell_text = []
for img, tpr, f1 in zip(images, tpr_scores, f1_scores):
    cell_text.append([img, f"{tpr:.3f}", f"{f1:.3f}"])
# Add average row
cell_text.append(["Average", f"{avg_tpr:.3f}", f"{avg_f1:.3f}"])

# Create figure and table
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')

columns = ["Image Name", "TPR", "F1-score"]
table = ax.table(
    cellText=cell_text,
    colLabels=columns,
    cellLoc='center',
    loc='center'
)

# Make header bold
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold')
    # Optional: make average row bold
    if i == len(cell_text):
        cell.set_text_props(weight='bold')

# Adjust column widths
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)  # scale width and height

plt.show()
