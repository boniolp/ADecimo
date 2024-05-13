description_intro = f"""
# Model Selection for Anomaly Detection in Time Series

Anomaly detection is a fundamental task for time-series analytics with important implications for the downstream performance of many applications. However, despite increasing academic interest and the large number of methods proposed in the literature, recent benchmark and evaluation studies demonstrated that no overall best anomaly detection methods exist when applied to very heterogeneous time series datasets. Therefore, the only scalable and viable solution to solve anomaly detection over very different time series collected from different domains is to propose a model selection method that will select, based on time series characteristics, the best anomaly detection method to run. Thus, this paper proposes a new pipeline for model selection based on time series classification and an extensive experimental evaluation of existing classification algorithms for this new pipeline. Our results demonstrate that model selection methods outperform every single anomaly detection method while being in the same order of magnitude regarding execution time.

Github repo: https://github.com/boniolp/MSAD

## Contributors

* [Paul Boniol](https://boniolp.github.io/paulboniol/) (Université Paris Cité)
* [Emmanouil Sylligardos](https://www.linkedin.com/in/emmanouil-sylligardos/) (École Normale Supérieure)
* [John Paparrizos](https://www.paparrizos.org/) (Ohio State University)
* [Panos Trahanias](https://www.linkedin.com/in/panos-trahanias-844bba108/?originalSubdomain=gr) (ICS-FORTH)
* [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/home.html) (Université Paris Cité)

## Datasets and Models

We host the datasets and pre-trained models at the following locations. You may find more details oin the dataset and the models in the corresponding tabs.
Moreover, you can download the datasets and the models using the following links:

- datasets: https://drive.google.com/file/d/1PQKwu5lZTnTmsmms1ipko9KF712Oc5DE/view?usp=share_link
- models: https://drive.google.com/file/d/1sQeqVZSBUuaJrvsueLtjLhACbARFS2dx/view?usp=sharing

## Model Selection Pipeline

We propose a benchmark and an evaluation of 16 time series classifiers used as model selection methods (with 12 anomaly detectors to be selected) applied on 16 datasets from different domains. Our pipeline can be summarized in the following figure.
"""


list_measures = ['VUS_PR', 'AUC_PR']

list_length = [16, 32, 64, 128, 256, 512, 768, 1024]

dataset_stats_real = ['period_length', 'ratio', 'nb_anomaly', 'average_anom_length',
	   'median_anom_length', 'std_anom_length', 'data_len']

dataset_cat = ['dataset', 'point_anom', 'seq_anom', 'mixed_anom', 'type_an']

dataset_stats = dataset_cat + dataset_stats_real

methods = ['inception_time_default_16',
	   'inception_time_default_32', 'inception_time_default_64', 'inception_time_default_128', 'inception_time_default_256',
	   'inception_time_default_512', 'inception_time_default_768', 'inception_time_default_1024', 'convnet_default_16',
	   'convnet_default_32', 'convnet_default_64', 'convnet_default_128', 'convnet_default_256', 'convnet_default_512',
	   'convnet_default_768', 'convnet_default_1024', 'resnet_default_16', 'resnet_default_32', 'resnet_default_64',
	   'resnet_default_128', 'resnet_default_256', 'resnet_default_512', 'resnet_default_768', 'resnet_default_1024',
	   'sit_conv_patch_16', 'sit_conv_patch_32', 'sit_conv_patch_64', 'sit_conv_patch_128', 'sit_conv_patch_256',
	   'sit_conv_patch_512', 'sit_conv_patch_768', 'sit_conv_patch_1024', 'sit_linear_patch_16', 'sit_linear_patch_32',
	   'sit_linear_patch_64', 'sit_linear_patch_128', 'sit_linear_patch_256', 'sit_linear_patch_512', 'sit_linear_patch_768',
	   'sit_linear_patch_1024', 'sit_stem_original_16', 'sit_stem_original_32', 'sit_stem_original_64', 'sit_stem_original_128',
	   'sit_stem_original_256', 'sit_stem_original_512', 'sit_stem_original_768', 'sit_stem_original_1024', 'sit_stem_relu_16',
	   'sit_stem_relu_32', 'sit_stem_relu_64', 'sit_stem_relu_128', 'sit_stem_relu_256', 'sit_stem_relu_512', 'sit_stem_relu_768',
	   'sit_stem_relu_1024', 'rocket_16', 'rocket_32', 'rocket_64', 'rocket_128', 'rocket_256', 'rocket_512', 'rocket_768', 'rocket_1024',
	   'ada_boost_16', 'ada_boost_32', 'ada_boost_64', 'ada_boost_128', 'ada_boost_256', 'ada_boost_512', 'ada_boost_768', 'ada_boost_1024',
	   'knn_16', 'knn_32', 'knn_64', 'knn_128', 'knn_256', 'knn_512', 'knn_768', 'knn_1024',
	   'decision_tree_16', 'decision_tree_32', 'decision_tree_64', 'decision_tree_128', 'decision_tree_256', 'decision_tree_512', 'decision_tree_768', 'decision_tree_1024',
	   'random_forest_16', 'random_forest_32', 'random_forest_64', 'random_forest_128', 'random_forest_256', 'random_forest_512', 'random_forest_768', 'random_forest_1024',
	   'mlp_16', 'mlp_32', 'mlp_64', 'mlp_128', 'mlp_256', 'mlp_512', 'mlp_768', 'mlp_1024',
	   'bayes_16', 'bayes_32', 'bayes_64', 'bayes_128', 'bayes_256', 'bayes_512', 'bayes_768', 'bayes_1024',
	   'qda_16', 'qda_32', 'qda_64', 'qda_128', 'qda_256', 'qda_512', 'qda_768', 'qda_1024',
	   'svc_linear_16', 'svc_linear_32', 'svc_linear_64', 'svc_linear_128', 'svc_linear_256', 'svc_linear_512', 'svc_linear_768', 'svc_linear_1024']

old_method = ['IFOREST', 'LOF', 'MP', 'NORMA', 'IFOREST1', 'HBOS', 'OCSVM',
	   'PCA', 'AE', 'CNN', 'LSTM', 'POLY']
	   
oracle = ['GENIE', 'MORTY']

methods_ens = [
	'inception_time_default_{}_score',
	'convnet_default_{}_score',
	'resnet_default_{}_score',
	'sit_conv_patch_{}_score',
	'sit_linear_patch_{}_score',
	'sit_stem_original_{}_score',
	'sit_stem_relu_{}_score',
	'rocket_{}_score',
	'ada_boost_{}_score',
	'knn_{}_score',
	'decision_tree_{}_score',
	'random_forest_{}_score',
	'mlp_{}_score',
	'bayes_{}_score',
	'qda_{}_score',
	'svc_linear_{}_score']

methods_conv = [
	'inception_time_default_{}_score',
	'convnet_default_{}_score',
	'resnet_default_{}_score', ]

methods_sit = [
	'sit_conv_patch_{}_score',
	'sit_linear_patch_{}_score',
	'sit_stem_original_{}_score',
	'sit_stem_relu_{}_score', ]

methods_ts = ['rocket_{}_score']

color_palette = [
	'#D32F2F', # Red
	'#C2185B', # Pink
	'#7B1FA2', # Purple
	'#512DA8', # Deep Purple
	'#303F9F', # Indigo
	'#1976D2', # Blue
	'#0288D1', # Light Blue
	'#0097A7', # Cyan
	'#00796B', # Teal
	'#388E3C', # Green
	'#689F38', # Light Green
	'#AFB42B', # Lime
	'#FBC02D', # Yellow
	'#FFA000', # Amber
	'#F57C00', # Orange
	'#E64A19', # Deep Orange
]


methods_classical = [
	'ada_boost_{}_score',
	'knn_{}_score',
	'decision_tree_{}_score',
	'random_forest_{}_score',
	'mlp_{}_score',
	'bayes_{}_score',
	'qda_{}_score',
	'svc_linear_{}_score']


all_datasets = ['SMD', 'NAB', 'SVDB', 'SensorScope', 'GHL', 'Genesis', 'OPPORTUNITY', 'ECG', 'MGAB', 'YAHOO', 'KDD21', 'IOPS', 'Daphnet', 'MITDB']



method_group = {
	   'Transformer': methods_sit,
	   'Convolutional': methods_conv,
	   'Rocket': methods_ts,
	   'Features': methods_classical}

template_names = {
	'inception_time_{}': 'InceptTime-{}',
	'convnet_{}': 'ConvNet-{}',
	'resnet_{}': 'ResNet-{}',
	'sit_conv_{}': 'SiT-conv-{}',
	'sit_linear_{}': 'SiT-linear-{}',
	'sit_stem_{}': 'SiT-stem-{}',
	'sit_stem_relu_{}': 'SiT-stem-ReLU-{}',
	'rocket_{}': 'Rocket-{}',
	'ada_boost_{}': 'AdaBoost-{}',
	'knn_{}': 'kNN-{}',
	'decision_tree_{}': 'DecisionTree-{}',
	'random_forest_{}': 'RandomForest-{}',
	'mlp_{}': 'MLP-{}',
	'bayes_{}': 'Bayes-{}',
	'qda_{}': 'QDA-{}',
	'svc_linear_{}': 'SVC-{}',
	'IFOREST': 'IForest',
	'LOF': 'LOF',
	'MP': 'MP',
	'NORMA': 'NormA',
	'IFOREST1': 'IForest1',
	'HBOS': 'HBOS',
	'OCSVM': 'OCSVM',
	'PCA': 'PCA',
	'AE': 'AE',
	'CNN': 'CNN',
	'LSTM': 'LSTM',
	'POLY': 'POLY',
	'Avg Ens': 'Avg Ens',
	'Oracle': 'Oracle',
	'best_ms': 'Best MS',
	'VUS_PR': 'VUS-PR',
	'label': 'Label',
	'best_ms': 'Best MS',
	'convnet': 'ConvNet',
	'resnet': 'ResNet',
	'rocket': 'Rocket',
	'knn': 'kNN',
	'sit_stem': 'SiT-stem',
	'sit': 'SiT',
	'feature_based': 'Feature-based',
	'AUC_PR': 'AUC-PR',
	'period_length': 'Period Length',
	'ratio': 'Anomaly Ratio',
	'nb_anomaly': 'Number of Anomalies',
	'average_anom_length': 'Average Anomaly Length',
	'median_anom_length': 'Median Anomaly Length',
	'std_anom_length': 'Standard Deviation of Anomaly Length',
	'data_len': 'Data Length',
}

# Set up methods' colors
methods_colors = {
	"oracle": "#FFFFFF",
	"avg_ens": "#FF7133",
	"best_ms": "#33D4FF",
	"detectors": "#CCCCCC",
	"feature_based": "#91AAC2",
	"sit": "#FFB522",
	"conv": "#4494FF",
	"conv_2": "#0048FF",
	"rocket": "#EA7DFF",
	"best_ad_train": "#228B22",
	"worst_ad_test": "#8B0000",
}



text_description_dataset = f"""
We use the time series in the TSB-UAD benchmark (16 public datasets from heterogeneous domains).
Briefly, TSB-UAD includes the following datasets:

| Dataset    | Description|
|:--|:---------:|
|Dodgers| is a loop sensor data for the Glendale on-ramp for the 101 North freeway in Los Angeles, and the anomalies represent unusual traffic after a Dodgers game.|
|ECG| is a standard electrocardiogram dataset, and the anomalies represent ventricular premature contractions. We split one long series (MBA_ECG14046) with length ∼ 1e7) into 47 series by first identifying the periodicity of the signal.|
|IOPS| is a dataset with performance indicators that reflect the scale, quality of web services, and health status of a machine.|
|KDD21| is a composite dataset released in a recent SIGKDD 2021 competition with 250 time series.|
|MGAB| is composed of Mackey-Glass time series with non-trivial anomalies. Mackey-Glass time series exhibit chaotic behavior that is difficult for the human eye to distinguish.|
|NAB| is composed of labeled real-world and artificial time series including AWS server metrics, online advertisement clicking rates, real time traffic data, and a collection of Twitter mentions of large publicly-traded companies.|
|SensorScope| is a collection of environmental data, such as temperature, humidity, and solar radiation, collected from a typical tiered sensor measurement system.|
|YAHOO| is a dataset published by Yahoo labs consisting of real and synthetic time series based on the real production traffic to some of the Yahoo production systems.|
|Daphnet| contains the annotated readings of 3 acceleration sensors at the hip and leg of Parkinson’s disease patients that experience freezing of gait (FoG) during walking tasks.|
|GHL| is a Gasoil Heating Loop Dataset and contains the status of 3 reservoirs, such as the temperature and level. Anomalies indicate changes in max temperature or pump frequency.|
|Genesis| is a portable pick-and-place demonstrator which uses an air tank to supply all the gripping and storage units.|
|MITDB| contains 48 half-hour excerpts of two-channel ambulatory ECG recordings obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979.|
|OPPORTUNITY (OPP)| is a dataset devised to benchmark human activity recognition algorithms (e.g., classification, automatic data segmentation, sensor fusion, and feature extraction). The dataset comprises the readings of motion sensors recorded while users executed typical daily activities.|
|Occupancy| contains experimental data used for binary classification (room occupancy) from temperature, humidity, light, and CO2. Ground-truth occupancy was obtained from time-stamped pictures that were taken every minute.|
|SMD (Server Machine Dataset)| is a 5-week-long dataset collected from a large Internet company. This dataset contains 3 groups of entities from 28 different machines.|
|SVDB| includes 78 half-hour ECG recordings chosen to supplement the examples of supraventricular arrhythmias in the MIT-BIH Arrhythmia Database.|


"""
text_description_AD=f"""
We use 12 anomaly detection methods proposed for univariate time series. The following table lists and describes the methods considered:

| Anomaly Detection Method    | Description|
|:--|:---------:|
|Isolation Forest (IForest) | This method constructs the binary tree based on space splitting, and the nodes with shorter path lengths to the root are more likely to be anomalies. |
|The Local Outlier Factor (LOF)| This method computes the ratio of the neighboring density to the local density. |
|The Histogram-based Outlier Score (HBOS)| This method constructs a histogram for the data and the inverse of the height of the bin is used as the outlier score of the data point. |
|Matrix Profile (MP)| This method calculates as anomaly the subsequence with the most significant 1-NN distance. |
|NORMA| This method identifies the normal pattern based on clustering and calculates each point's effective distance to the normal pattern. |
|Principal Component Analysis (PCA)| This method projects data to a lower-dimensional hyperplane, and data points with a significant distance from this plane can be identified as outliers. |
|Autoencoder (AE)|This method projects data to the lower-dimensional latent space and reconstructs the data, and outliers are expected to have more evident reconstruction deviation. |
|LSTM-AD| This method build a non-linear relationship between current and previous time series (using Long-Short-Term-Memory cells), and the outliers are detected by the deviation between the predicted and actual values. |
|Polynomial Approximation (POLY)| This method build a non-linear relationship between current and previous time series (using polynomial decomposition), and the outliers are detected by the deviation between the predicted and actual values. |
| CNN | This method build a non-linear relationship between current and previous time series (using convolutional Neural Network), and the outliers are detected by the deviation between the predicted and actual values. |
|One-class Support Vector Machines (OCSVM)| This method fits the dataset to find the normal data's boundary. |

You may find more details (and the references) in the TSB-UAD benchmark [paper](https://www.paparrizos.org/papers/PaparrizosVLDB22b.pdf).
"""

text_description_MS = f"""
We consider 16 time series classification (TSC) algorithms used as model selection. The following table lists and describes the methods considered:

| TSC  (as model seleciton)  | Description|
|:--|:---------:|
| SVC | maps training examples to points in space to maximize the gap between the two categories. |
| Bayes | uses Bayes’ theorem to predict the class of a new data point using the posterior probabilities for each class. |
| MLP | consists of multiple layers of interconnected neurons. |
| QDA | is a discriminant analysis algorithm for classification problems. |
| Adaboost | is a meta-algorithm using boosting technique with weak classifiers. |
| Decision Tree | is a tree-based approach that splits data points into different leaves based on features. |
| Random Forest  | is an ensemble Decision Trees fed with random samples (with replacement) of the training set and random set of features. |
| kNN | assigns the most common class among its k nearest neighbors. |
| Rocket | transforms input time series using a small set of convolutional kernels, and uses the transformed features to train a linear classifier. |
| ConvNet  | uses convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. |
| ResNet | is a ConvNet with residual connections between convolutional block. |
| InceptionTime | is a combination of ResNets with kernels of multiple sizes. |
| SIT-conv | is a transformer architecture with a convolutional layer as input. |
| SIT-linear | is a transformer architecture for which the time series is divided into non-overlapping patches and linearly projected into the embedding space. |
| SIT-stem | is a transformer architecture with convolutional layers with increasing dimensionality as input. |
| SIT-stem-ReLU | is similar to SIT-stem but with Scaled ReLU. |
"""

