# 3D DenseNet for classification and characterisation of proximal humerus fractures on CT-scans
## Tasks
Proximal humerus CT-scans will be classified on the following characteristics: 

-  Fracture parts (1-part, 2-part, multipart)
- 	Glenohumeral dislocation (yes/no)
- 	Greater tuberosity displacement ≥ 1 cm (yes/no)
- 	Head-shaft displacement (≥ 25% overlap = none, 25 - 5% = substantial, ≤ 5% = complete)
- 	Varus deformation ≤ 100° (yes/no)
- 	Articular involvement(none or rim, subcritical, headsplit)

### Characteristics -> labels
1. Fracture parts -> fracture_classification
2. Glenohumeral dislocation -> fracture_classification(dislocation)
3. Greater tuberosity displacement ≥ 1 cm -> gt_displacement_greater_equal_to_1cm
4. Head-shaft displacement -> shaft_translation
5. Varus deformation -> varus_malalignment
6. Articular involvement -> art_involvement

## Environment

- Python: 3.x
- PyTorch: 2.7.0+cu128
- CUDA: 12.8

```
pip install -r requirements.txt
```

## Data placement (required)

Put your data like this :
```
your_project/
├─ V3.ipynb     
├─ v3_project
├─ run.py
├─ train_labels.csv
├─ test_labels.csv
└─ new_datasets/
   ├─ one_side_train/
   │  ├─ <case1>.nii
   │  ├─ <case2>.nii
   │  └─ ...
   └─ one_side_test/
      ├─ <caseX>.nii
      └─ ...
```


- CT files must be **.nii** (the code currently searches `*.nii`).

### Label CSV columns (required)
Both CSVs must include:

`patient_id, fracture_classification, gt_displacement_greater_equal_to_1cm, shaft_translation, varus_malalignment, art_involvement`

---

## Training

### Notebook
Open `V3.ipynb` and **Run All**.

Or

```
jupyter nbconvert --execute V3.ipynb --to notebook --output V3_executed.ipynb
```
then you can see the results in a new notebook `V3_executed.ipynb`.

### Python version
```bash
python run.py
```

## Test & Visulization

If you want to see the internal validation result, you can also use the notebooks in this repo:

- `Validation.ipynb`

And use the checkpoint below:

### Checkpoint
Use the pretrained checkpoint:
- `asset/final_multi_3d_clf4.pth`  
  [3dDenseNet121](https://github.com/AIML-MED/CT-Fracture/blob/FINAL-VERSION/asset/final_multi_3d_clf4.pth)
  
Please place this checkpoint in the same directory as `Validation.ipynb` before running validation or visualization.
