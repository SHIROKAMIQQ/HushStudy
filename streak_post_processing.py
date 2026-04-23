import pandas as pd
import numpy as np

INPUT_CSV = "tlc-apr-8-dataset2.csv"
OUTPUT_CSV = "tlc-apr-8-dataset2_streaked.csv"

df = pd.read_csv(INPUT_CSV)

# ensure correct order
df = df.sort_values("timestamp_start").reset_index(drop=True)

# -------------------------
# FILL CHATTER STREAK
# -------------------------
streak = []
current = 0

for val in df["is_chatter"]:
    if val == 1:
        current += 1
    else:
        current = 0
    streak.append(current)

df["chatter_streak"] = streak

# -------------------------
# LOG FEATURE
# -------------------------
df["streak_log"] = np.log1p(df["chatter_streak"])

# -------------------------
# SAVE
# -------------------------
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)