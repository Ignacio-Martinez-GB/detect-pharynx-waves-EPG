% Load data from Excel file
data = readtable('HeatMap2.xlsx'); % Load dataset from Excel file (replace with your own file)

% Define biomechanical variables and their display labels
variables = {'Area', 'Length', 'NormalizedSpeed', 'NormalizedWavelength', 'NormalizedWaveamplitude', 'BodyBends'}; % Column names in your table
variableLabels = {'Area', 'Body Length', 'Normalized Speed', 'Normalized Wavelength', 'Normalized Wave amplitude', 'Body bends per minute'}; % Labels shown on the heatmap

Strains = {'N2', 'ATM1', 'CZ26389', 'ATM1xCZ26389'}; % Desired strain order (edit this list based on your dataset)

% Preallocate matrix for normalized values
normalizedData = zeros(numel(Strains), numel(variables)); % Preallocate normalization matrix

% Normalize all strains relative to the control strain
controlCepa = 'N2'; % Set your control strain here
controlData = data(strcmp(data.Strain, controlCepa), :); % Extract rows for the control strain

for i = 1:numel(Strains)
    cepaData = data(strcmp(data.Strain, Strains{i}), :); % Extract rows for each strain
    for v = 1:numel(variables)
        % Compute log2 fold change relative to the control strain
        normalizedData(i, v) = log2(mean(cepaData{:, variables{v}}) / mean(controlData{:, variables{v}}));
    end
end

% Perform hierarchical clustering
distances = pdist(normalizedData, 'euclidean'); % Compute Euclidean distances between strains
linkages = linkage(distances, 'average'); % Perform hierarchical clustering using average linkage

% Keep dendrogram and heatmap in synchronized order
[~, dendrogramOrder] = ismember(Strains, Strains); % Keep fixed order (modify if you want auto-clustering order)

% Create heatmap using a green-to-red custom colormap
nColors = 256; % Number of levels in the colormap (higher = smoother gradient)

% Generate green-to-red colormap gradient (green → red)
greenToRedColormap = [linspace(0, 1, nColors)', linspace(1, 0, nColors)', zeros(nColors, 1)]; % Custom gradient

% Custom strain labels (edit depending on your strains) 
cepaLabels = {'WT', ...
    '\it{-mctp-1}', ...     
    '\it{-esyt-2}', ...     
    '-mctp-1/-esyt-2'};    

% Display dendrogram and heatmap within the same figure
figure;

% Subplot 1: Dendrogram plot
subplot(1, 2, 1);
dendrogram(linkages, 'Labels', cepaLabels(dendrogramOrder), 'Orientation', 'left'); % Plot dendrogram
xlabel('Euclidian distance'); % X-axis label
ylabel('Strain'); % Y-axis label

% Subplot 2: Heatmap visualization
subplot(1, 2, 2);
heatmap(variableLabels, cepaLabels, normalizedData(dendrogramOrder, :), ...
    'Colormap', greenToRedColormap, 'ColorLimits', [0, 0.5], 'CellLabelColor', 'none'); % Generate heatmap

colorbar; % Add colorbar for scale
