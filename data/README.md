# Facial Recognition Dataset

The FR experiments (Section 4.2 of the paper) expect two parquet files in this
folder. 

| File | Contents |
|---|---|
| `Scores.parquet` | One row per image pair - `image_1`, `image_2`, `score` (similarity), `diag` (1 = genuine / same identity, 0 = impostor) |
| `Qualities.parquet` | One row per image - `image` identifier + one column per covariate (image quality, resolution, hair color, age estimate, …; full list in Table 6 of the paper) |

Size: 125,052 genuine pairs and 1,149,498 impostor pairs. The dataset contains
only similarity scores and per-image covariates - **no images and no identity
information**.

Loading:

```python
import pandas as pd
scores    = pd.read_parquet("data/Scores.parquet",    engine="fastparquet")
qualities = pd.read_parquet("data/Qualities.parquet", engine="fastparquet")
```

