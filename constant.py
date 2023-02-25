list_measures = ['VUS_PR','AUC_PR','VUS_ROC','AUC_ROC']

list_length = [16,32,64,128,256,512,768,1024]

dataset_stats_real = ['period_length', 'ratio', 'nb_anomaly', 'average_anom_length',
       'median_anom_length', 'std_anom_length', 'data_len']

dataset_cat = ['dataset','point_anom', 'seq_anom', 'mixed_anom', 'type_an']

dataset_stats = dataset_cat + dataset_stats_real

methods = ['inception_time_default_16',
       'inception_time_default_32','inception_time_default_64','inception_time_default_128','inception_time_default_256',
       'inception_time_default_512','inception_time_default_768','inception_time_default_1024','convnet_default_16',
       'convnet_default_32','convnet_default_64','convnet_default_128','convnet_default_256','convnet_default_512',
       'convnet_default_768','convnet_default_1024','resnet_default_16','resnet_default_32','resnet_default_64',
       'resnet_default_128','resnet_default_256','resnet_default_512','resnet_default_768','resnet_default_1024',
       'sit_conv_patch_16','sit_conv_patch_32','sit_conv_patch_64','sit_conv_patch_128','sit_conv_patch_256',
       'sit_conv_patch_512','sit_conv_patch_768','sit_conv_patch_1024','sit_linear_patch_16','sit_linear_patch_32',
       'sit_linear_patch_64','sit_linear_patch_128','sit_linear_patch_256','sit_linear_patch_512','sit_linear_patch_768',
       'sit_linear_patch_1024','sit_stem_original_16','sit_stem_original_32','sit_stem_original_64','sit_stem_original_128',
       'sit_stem_original_256','sit_stem_original_512','sit_stem_original_768','sit_stem_original_1024','sit_stem_relu_16',
       'sit_stem_relu_32','sit_stem_relu_64','sit_stem_relu_128','sit_stem_relu_256','sit_stem_relu_512','sit_stem_relu_768',
       'sit_stem_relu_1024','rocket_16','rocket_32','rocket_64','rocket_128','rocket_256','rocket_512','rocket_768','rocket_1024',
       'ada_boost_16','ada_boost_32','ada_boost_64','ada_boost_128','ada_boost_256','ada_boost_512','ada_boost_768','ada_boost_1024',
       'knn_16','knn_32','knn_64','knn_128','knn_256','knn_512','knn_768','knn_1024',
       'decision_tree_16','decision_tree_32','decision_tree_64','decision_tree_128','decision_tree_256','decision_tree_512','decision_tree_768','decision_tree_1024',
       'random_forest_16','random_forest_32','random_forest_64','random_forest_128','random_forest_256','random_forest_512','random_forest_768','random_forest_1024',
       'mlp_16','mlp_32','mlp_64','mlp_128','mlp_256','mlp_512','mlp_768','mlp_1024',
       'bayes_16','bayes_32','bayes_64','bayes_128','bayes_256','bayes_512','bayes_768','bayes_1024',
       'qda_16','qda_32','qda_64','qda_128','qda_256','qda_512','qda_768','qda_1024',
       'svc_linear_16','svc_linear_32','svc_linear_64','svc_linear_128','svc_linear_256','svc_linear_512','svc_linear_768','svc_linear_1024']

old_method = ['IFOREST', 'LOF', 'MP', 'NORMA', 'IFOREST1', 'HBOS', 'OCSVM',
       'PCA', 'AE', 'CNN', 'LSTM', 'POLY']

oracle = ['GENIE','MORTY']

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
    'resnet_default_{}_score',]

methods_sit = [
    'sit_conv_patch_{}_score',
    'sit_linear_patch_{}_score',
    'sit_stem_original_{}_score',
    'sit_stem_relu_{}_score',]

methods_ts = ['rocket_{}_score']

methods_classical = [
    'ada_boost_{}_score',
    'knn_{}_score',
    'decision_tree_{}_score',
    'random_forest_{}_score',
    'mlp_{}_score',
    'bayes_{}_score',
    'qda_{}_score',
    'svc_linear_{}_score']


method_group = {
       'Transformer': methods_sit,
       'Convolutional': methods_conv,
       'Rocket': methods_ts,
       'Features': methods_classical}

description_intro = f"""
# MSAD: Model Selection for Anomaly Detection in Time Series

Anomaly detection is a fundamental task for time-series analytics with important implications for the downstream performance of many applications. However, despite increasing academic interest and the large number of methods proposed in the literature, recent benchmark and evaluation studies demonstrated that no overall best anomaly detection methods exist when applied to very heterogeneous time series datasets. Therefore, the only scalable and viable solution to solve anomaly detection over very different time series collected from different domains is to propose a model selection method that will select, based on time series characteristics, the best anomaly detection method to run. Thus, this paper proposes a new pipeline for model selection based on time series classification and an extensive experimental evaluation of existing classification algorithms for this new pipeline. Our results demonstrate that model selection methods outperform every single anomaly detection method while being in the same order of magnitude regarding execution time.


## Contributors

* Emmanouil Sylligardos (ICS-FORTH)
* Paul Boniol (Université Paris Cité)

## Datasets and Models

We host the datasets and pre-trained models at the following locations. You may find more details oin the dataset and the models in the corresponding tabs.
Moreover, you can download the datasets and the models using the following links:

- datasets: TODO
- models: TODO

## Model Selection Pipeline

We propose a benchmark and an evaluation of 16 time series classifiers used as model selection methods (with 12 anomaly detectors to be selected) applied on 16 datasets from different domains. Our pipeline can be summarized in the following figure.
"""


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


