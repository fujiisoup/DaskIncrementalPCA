# Incremental PCA with DASK

## Install
```bash
python setup.py develop
```

## Usage
```python
from incremental_pca import IncrementalPCA
model = IncrementalPCA(n_components=16, whiten=False,       copy=True,
                 batch_size=None, svd_solver='auto')
```