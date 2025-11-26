
import json

file_path = 'ModelTrain.ipynb'

with open(file_path, 'r') as f:
    data = json.load(f)

# Find the cell with the code to replace
target_source = [
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "\n",
    "trf = ColumnTransformer([\n",
    "    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])\n",
    "],\n",
    "remainder='passthrough')"
]

replacement_source = [
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "\n",
    "try:\n",
    "    encoder = OneHotEncoder(sparse_output=False, drop='first')\n",
    "except TypeError:\n",
    "    encoder = OneHotEncoder(sparse=False, drop='first')\n",
    "\n",
    "trf = ColumnTransformer([\n",
    "    ('trf', encoder, ['batting_team', 'bowling_team', 'city'])\n",
    "],\n",
    "remainder='passthrough')"
]

found = False
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        # Check if source matches exactly
        if cell['source'] == target_source:
            cell['source'] = replacement_source
            found = True
            break

if found:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=1)
    print("Notebook fixed successfully.")
else:
    print("Target code not found in notebook.")
    # Print first few lines of sources to debug if needed
    # for cell in data['cells']:
    #     if cell['cell_type'] == 'code' and 'ColumnTransformer' in ''.join(cell['source']):
    #         print("Found similar cell:")
    #         print(cell['source'])
