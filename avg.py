import json

# Load the uploaded JSON file
file_path = 'UniAD/test/base_track_map/Fri_Dec__6_19_05_59_2024/results_nusc.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract the `map_results` section
map_results = data.get("map_results", {})

# Initialize accumulators for IoU categories
iou_categories = ["drivable_iou", "lanes_iou", "divider_iou", "crossing_iou", "contour_iou"]
iou_totals = {key: 0.0 for key in iou_categories}
iou_counts = {key: 0 for key in iou_categories}

# Iterate through entries in `map_results` to collect IoU values
for entry in map_results.values():
    for category in iou_categories:
        if category in entry:  # Check if the IoU category exists in the entry
            iou_totals[category] += entry[category]
            iou_counts[category] += 1

# Compute averages
average_ious = {key: (iou_totals[key] / iou_counts[key] if iou_counts[key] > 0 else 0) for key in iou_categories}

# Display results
print("Average IoUs:")
for category, average in average_ious.items():
    print(f"{category}: {average:.4f}")
