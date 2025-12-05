# HeatMapHierarchicalClustering
MATLAB code for computing log₂-normalized biomechanical parameters across C. elegans strains and generating a hierarchical-clustered heatmap. The script loads an Excel file (“HeatMap.xlsx”), normalizes each parameter relative to wild type (N2), and visualizes phenotypic similarity using Euclidean distance and average-linkage clustering.

Data loaded from HeatMap.xlsx using readtable. Parameters grouped by strain (N2, ATM1, CZ26389, ATM1×CZ26389). Log₂ ratios computed relative to N2.
Pairwise distances computed using Euclidean distance. Hierarchical clustering performed using average linkage.

Visualization includes:
Dendrogram (left)
Heatmap with green-to-red colormap (right)


1.Open MATLAB.

2.Place HeatMap0312.m and HeatMap.xlsx in the same directory.

Example:
<img width="1763" height="339" alt="image" src="https://github.com/user-attachments/assets/2c3f40b0-1c71-4e9d-87cd-6a1bb5ed940b" />


3.Run:HeatMap0312

4.A dendrogram + heatmap figure will be generated.
