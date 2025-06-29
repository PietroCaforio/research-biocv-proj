import json

import numpy as np

performance = json.load(
    open("./CPTACPDA_mixed5_UNI2_MedSAM2/cv_results_multival.json")
)


ct_missing_values = np.array(
    [
        performance["fold_results"][i]["best_monitor_values"]["ct_missing"]
        for i in range(4)
    ]
)
histo_missing_values = np.array(
    [
        performance["fold_results"][i]["best_monitor_values"]["histo_missing"]
        for i in range(4)
    ]
)
mixed_missing_values = np.array(
    [
        performance["fold_results"][i]["best_monitor_values"]["mixed_missing"]
        for i in range(4)
    ]
)

print(
    f"ct_missing c-index: {ct_missing_values.mean():.6f} ± {ct_missing_values.std():.6f}"
)
print(
    f"histo_missing c-index: {histo_missing_values.mean():.6f} ± {histo_missing_values.std():.6f}"
)
print(
    f"mixed_missing c-index: {mixed_missing_values.mean():.6f} ± {mixed_missing_values.std():.6f}"
)
