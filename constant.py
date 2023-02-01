list_measures = ['VUS_PR','AUC_PR','VUS_ROC','AUC_ROC']

list_length = [16,32,64,128,256,512,768,1024]

dataset_stats = ['dataset','period_length', 'ratio', 'nb_anomaly', 'average_anom_length',
       'median_anom_length', 'std_anom_length', 'data_len',
       'point_anom', 'seq_anom', 'mixed_anom', 'type_an']

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
