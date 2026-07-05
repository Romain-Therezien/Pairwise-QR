# Dataset README

This README describes the dataset provided as supplementary material for our UAI submission. 
The dataset contains information about image pairs and image quality measures used in our experiments.

## Files

1. **Scores.txt**  
   - Contains a list of image pairs with their associated scores.  
   - Columns:
     - `image1`, `image2`: Identifiers of the two images in the pair.
     - `score` : Numeric similarity score between the images.
     - `diag`: 1 if the images are of the same identity, 0 otherwise.  

2. **Qualities.txt**  
   - Contains all images listed with their associated quality measures.  
   - Columns:
     - `image`: Identifier of the image.
     - Additional columns: Various quality measures (one column per measure). 

## Notes
- Example of loading the files in Python with pandas:
```python
import pandas as pd
scores = pd.read_parquet('Scores.parquet', engine='fastparquet')
qualities = pd.read_parquet('Qualities.parquet', engine='fastparquet')
```

# Code README
This README describes the code provided as supplementary material for our UAI submission. 
The code is organized in a Jupyter notebook named `main.ipynb` and contains the following sections:
1. **Data Loading**: Code to load the `Scores` and `Qualities` datasets.
2. **Data Preprocessing**: Code to merge the datasets and prepare the features and target variable for modeling.
3. **Model Training**: Code to train regression models (e.g., neural networks) to predict the similarity scores based on the quality measures.
4. **SHAP Analysis**: Code to compute SHAP values for the trained models and analyze the impact of different features across score levels.
5. **Visualization**: Code to visualize the SHAP values and their impact on the predicted scores.
- The notebook includes functions to compute the impact of features across score levels and to visualize these impacts using line plots. 
- The code is structured to allow easy modification and experimentation with different models, features, and visualization techniques.