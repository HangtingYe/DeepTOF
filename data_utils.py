import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
from torch.utils.data import TensorDataset, DataLoader


from config import feat_cols_20, feat_cols_30, feat_cols_40, feat_cols_50, feat_cols_60, feat_cols_70, feat_cols_all

feat_cols_dict = {'20':feat_cols_20, '30':feat_cols_30, '40':feat_cols_40, '50':feat_cols_50, '60':feat_cols_60, '70':feat_cols_70, 'all':feat_cols_all}

import warnings
# warnings.simplefilter("error")

id_column = 'PID'


target_column = ['bp', 'pf']

forecast_horizons = ['{}Y'.format(i + 1) for i in range(8)]
nb_years = len(forecast_horizons)

num_target_columns = len(target_column) * nb_years * 2

hsq_column = ['HSQ{:02d}'.format(i + 1) for i in range(36)]

num_hsq_columns = len(hsq_column) * nb_years * 2


def load_raw_data():
    data = pd.read_csv('./data/SPORTDATA8yr.csv', low_memory=False)

    obs_ID = data[data['status'] == 'OBS'].PID.unique()
    rct_ID = data[data['status'] == 'RCT'].PID.unique()

    return data.set_index(id_column), obs_ID, rct_ID


def surgical_feature_from_data(data, feat_cols):
    def _parse_input(x):
        x = x.sort_values(by='fu_days')
        x = x[x.ATX == 'Treatment B'].iloc[-1]
        return x[feat_cols]

    def _parse_target(x):
        return x[target_column]

    def _parse_hsq(x):
        return x[hsq_column]

    inputs = data.groupby(level=0).apply(_parse_input)

    _data = data[data.ATX == 'Treatment A']

    for d in forecast_horizons:
        targets = _data[_data.survey_type == d]\
            .groupby(level=0).apply(_parse_target)
        targets.columns = ['surgical_{}_{}'
                           .format(x, d) for x in targets]

        inputs = inputs.join(targets, how='left')

    for d in forecast_horizons:
        hsq = _data[_data.survey_type == d]\
            .groupby(level=0).apply(_parse_hsq)
        hsq.columns = ['surgical_{}_{}'
                       .format(x, d) for x in hsq]

        inputs = inputs.join(hsq, how='left')

    return inputs


def non_surgical_feature_from_data(data, feat_cols):
    def _parse_input(x):
        x = x.sort_values(by='fu_days')
        # ensure the first row is baseline
        assert(x.iloc[0].survey_type == 'BA')
        x = x.iloc[0]
        return x[feat_cols]

    def _parse_target(x):
        return x[target_column]

    def _parse_hsq(x):
        return x[hsq_column]

    inputs = data.groupby(level=0).apply(_parse_input)

    _data = data[data.ATX == 'Treatment B']

    for d in forecast_horizons:
        targets = _data[_data.survey_type == d]\
            .groupby(level=0).apply(_parse_target)
        targets.columns = ['non_surgical_{}_{}'
                           .format(x, d) for x in targets]

        inputs = inputs.join(targets, how='left')

    for d in forecast_horizons:
        hsq = _data[_data.survey_type == d]\
            .groupby(level=0).apply(_parse_hsq)
        hsq.columns = ['non_surgical_{}_{}'
                       .format(x, d) for x in hsq]

        inputs = inputs.join(hsq, how='left')

    return inputs


def parse_inputs_from_data(data, feat_cols):
    nsur_inputs = non_surgical_feature_from_data(data, feat_cols)
    sur_inputs = surgical_feature_from_data(data, feat_cols)

    inputs = pd.concat([nsur_inputs, sur_inputs], axis=0)

    surgical = [0] * nsur_inputs.shape[0] + [1] * sur_inputs.shape[0]
    inputs['surgical'] = surgical

    return inputs


def parse_normalizer(train_feature, test_feature):
    normalizers = dict()

    for key in train_feature:
        normalizers[key] = []

        cate_data = pd.concat(
            [
                train_feature[key], test_feature[key]
            ], axis=0
        )

        # for numerical features, we only use the training part to avoid the data leak
        num_data = train_feature[key]

        for col in num_data.columns:
            try:
                num_data[col] = num_data[col].astype('float')
                norm_ = StandardScaler()
                norm_.fit(num_data[col].values.reshape(-1, 1))
            except Exception as e:
                cate_data[col] = cate_data[col].astype('str')
                norm_ = LabelEncoder()
                norm_.fit(cate_data[col].values.reshape(-1, ))

            normalizers[key].append(norm_)

    return normalizers


def _apply_normalizer(data_dict, normalizers):
    for key in data_dict:
        if key == 'surgical':
            continue

        data = data_dict[key]
        norm_ = normalizers[key]

        for i, col in enumerate(data.columns):
            if isinstance(norm_[i], StandardScaler):
                data[col] = data[col].astype('float')
                data[col] = norm_[i].transform(
                    data[col].values.reshape(-1, 1)
                )
            else:
                data[col] = data[col].astype('str')
                data[col] = norm_[i].transform(
                    data[col].values.reshape(-1, )
                )

    return data_dict


def _check_column_type(x):
    if x.startswith('non_surgical_') or x.startswith('surgical_'):
        if x.find('HSQ') != -1:
            return 'hsq'
        else:
            return 'target'
    elif x == 'surgical':
        return 'surgical'
    else:
        return 'feature'

    raise ValueError


def _decompose(data):
    feature = data[[
        x for x in data.columns if _check_column_type(x) == 'feature'
    ]]
    target = data[[
        x for x in data.columns if _check_column_type(x) == 'target'
    ]]
    hsq = data[[
        x for x in data.columns if _check_column_type(x) == 'hsq'
    ]]
    surgical = data[[
        x for x in data.columns if _check_column_type(x) == 'surgical'
    ]]

    return {
        'feature': feature,
        'target': target,
        'hsq': hsq,
        'surgical': surgical
    }


def _rm_all_nan(data):
    target = data[[
        x for x in data.columns if _check_column_type(x) == 'target'
    ]]

    mask = (~pd.isnull(target)).sum(axis=1) != 0

    return data[mask]


def load_feature(train_data, test_data, args):
    print('Train data shape: ', train_data.shape)
    print('Test data shape: ', test_data.shape)

    feat_cols = feat_cols_dict[args.feature]

    train_feature = parse_inputs_from_data(train_data, feat_cols)
    test_feature = parse_inputs_from_data(test_data, feat_cols)

    train_feature = _rm_all_nan(train_feature)
    test_feature = _rm_all_nan(test_feature)

    train_feature = _decompose(train_feature)
    test_feature = _decompose(test_feature)

    normalizers = parse_normalizer(train_feature, test_feature)

    train_feature = _apply_normalizer(train_feature, normalizers)
    test_feature = _apply_normalizer(test_feature, normalizers)

    print('Train feature shape: ', train_feature['feature'].shape)
    print('Test feature shape: ', test_feature['feature'].shape)

    return train_feature, test_feature, normalizers


def build_data_loader(data, batch_size=128, shuffle=False):
    feature = data['feature'].fillna(0.0)
    target = data['target']
    hsq = data['hsq']
    surgical = data['surgical']

    feature = torch.from_numpy(feature.values)
    target = torch.from_numpy(target.values)
    hsq = torch.from_numpy(hsq.values)
    surgical = torch.from_numpy(surgical.values)

    dataset = TensorDataset(feature, target, hsq, surgical)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
