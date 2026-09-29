scores = [23, 18, 82, 90, 55]
current_difficulty = 3
consecutive_high = 0
consecutive_low = 0

for i, score in enumerate(scores):
    if score >= 75:
        consecutive_high += 1
        consecutive_low = 0
    elif score < 40:
        consecutive_low += 1
        consecutive_high = 0
    else:
        consecutive_high = 0
        consecutive_low = 0

    if consecutive_high >= 2 and current_difficulty < 5:
        current_difficulty += 1
        consecutive_high = 0
    elif consecutive_low >= 2 and current_difficulty > 1:
        current_difficulty -= 1
        consecutive_low = 0

    print(f"Turn {i+1}: score={score} -> difficulty={current_difficulty}")
    

