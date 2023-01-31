methods = ['convnet_default_16', 'convnet_default_32',
       'convnet_default_64', 'convnet_default_128', 'convnet_default_256',
       'convnet_default_512', 'convnet_default_768', 'convnet_default_1024',
       'inception_time_default_16', 'inception_time_default_32',
       'inception_time_default_64', 'inception_time_default_128',
       'inception_time_default_256', 'inception_time_default_512',
       'inception_time_default_768', 'inception_time_default_1024',
       'resnet_default_16', 'resnet_default_32', 'resnet_default_64',
       'resnet_default_128', 'resnet_default_256', 'resnet_default_512',
       'resnet_default_768', 'resnet_default_1024', 'sit_conv_patch_16',
       'sit_conv_patch_32', 'sit_conv_patch_64', 'sit_conv_patch_128',
       'sit_conv_patch_256', 'sit_conv_patch_512', 'sit_conv_patch_768',
       'sit_conv_patch_1024', 'sit_linear_patch_16', 'sit_linear_patch_32',
       'sit_linear_patch_64', 'sit_linear_patch_128', 'sit_linear_patch_256',
       'sit_linear_patch_512', 'sit_linear_patch_768', 'sit_linear_patch_1024',
       'sit_stem_original_16', 'sit_stem_original_32', 'sit_stem_original_64',
       'sit_stem_original_128', 'sit_stem_original_256',
       'sit_stem_original_512', 'sit_stem_original_768',
       'sit_stem_original_1024', 'sit_stem_relu_16', 'sit_stem_relu_32',
       'sit_stem_relu_64', 'sit_stem_relu_128', 'sit_stem_relu_256',
       'sit_stem_relu_512', 'sit_stem_relu_768', 'sit_stem_relu_1024','rocket_16', 'rocket_32',
       'rocket_64', 'rocket_128', 'rocket_256', 'rocket_512', 'rocket_768',
       'rocket_1024', 'GENIE',
       'MORTY', 'IFOREST', 'LOF', 'MP', 'NORMA', 'IFOREST1', 'HBOS', 'OCSVM',
       'PCA', 'AE', 'CNN', 'LSTM', 'POLY']

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
