# Hyper-Parameters for Clustering

Total number of samples should be set to **32768**.



## UMAP

### For Clustering

| Model        | total_vis_sample_num | n_neighbors | min_dist | n_components |
|--------------|----------------------|-------------|----------|--------------|
| GraphsGPT-1W | 32768                | 40          | 0.05     | 2            |
| GraphsGPT-2W | 32768                | 40          | 0.05     | 2            |
| GraphsGPT-4W | 32768                | 100         | 0.05     | 2            |
| GraphsGPT-8W | 32768                | 100         | 0.05     | 2            |

### For Visualization

| Model        | total_vis_sample_num | n_neighbors | min_dist | n_components |
|--------------|----------------------|-------------|----------|--------------|
| GraphsGPT-1W | 32768                | 40          | 0.8      | 2            |
| GraphsGPT-2W | 32768                | 40          | 0.8      | 2            |
| GraphsGPT-4W | 32768                | 40          | 0.7      | 2            |
| GraphsGPT-8W | 32768                | 40          | 0.7      | 2            |



## HDBSCAN

| Model        | min_cluster_size | min_samples | cluster_selection_epsilon | alpha |
|--------------|------------------|-------------|---------------------------|-------|
| GraphsGPT-1W | 48               | 64          | 0.25                      | 1.0   |
| GraphsGPT-2W | 48               | 64          | 0.25                      | 1.0   |
| GraphsGPT-4W | 32               | 48          | 0.2                       | 1.0   |
| GraphsGPT-8W | 32               | 48          | 0.2                       | 1.0   |



