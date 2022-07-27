

import json

with open("annotations_cocofied_small.json") as f:
    d = json.load(f)
    cat = d["categories"]

with open("alfred_categories_mapping_small.json", "w+") as f:
    json.dump(cat, f)
