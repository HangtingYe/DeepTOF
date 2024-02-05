import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import ipdb
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import pandas as pd
import pickle
import random
from config import feasible_bp, feasible_pf
from transformer_harv import Model
from data_utils import load_raw_data, load_feature, build_data_loader, nb_years

from sklearn.model_selection import KFold

feasible_bp = np.asarray(feasible_bp)
feasible_pf = np.asarray(feasible_pf)

nb_feature = 70

batch_size = 128


#loss_func = F.l1_loss
def loss_func(pred, label, mask):
   return (_masked_loss((pred - label)**2, mask)+1e-2)**0.5/((_masked_loss(label**2, mask)+1e-2)**0.5)
# def loss_func(pred, label, mask):
#     return torch.mean((-1 * pearsonr(pred, label)))
# def loss_func(pred, label, mask):
#     #ipdb.set_trace()
#     mask = 1 - mask.float()
#     pred_mean = (torch.sum(pred * mask, dim=-1) / (torch.sum(mask, dim=-1)+1e-2)).reshape(-1, 1)
#     label_mean = (torch.sum(label * mask, dim=-1) / (torch.sum(mask, dim=-1)+1e-2)).reshape(-1, 1)
#     a = torch.sum(((pred-pred_mean)*mask) * ((label-label_mean)*mask), dim=-1)
#     b = torch.sqrt(torch.sum(((pred-pred_mean)*mask)**2, dim=-1))
#     c = torch.sqrt(torch.sum(((label-label_mean)*mask)**2, dim=-1))
#     return  torch.mean(-1 * a/((b+1e-2)*(c+1e-2)))
    

def _masked_loss(loss, mask):
    mask = 1 - mask.float()
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def _select_loss(selector, nb_feature=nb_feature):
    x = torch.cat(list(selector), dim=0)
    x = torch.clamp(x, 0, 1)

    l1 = torch.sum(x.abs())
    l2 = torch.sum(x ** 2)

    loss_sparse = l1 / l2
    loss_constraint = torch.abs(l1 - nb_feature) / nb_feature

    return loss_sparse + loss_constraint * 2


def calc_hsq(hsq, normalizers):
    def _get_stat(normalizer):
        mean, std = [], []
        for norm_ in normalizer:
            mean.append(norm_.mean_[0])
            std.append(norm_.scale_[0])
        mean = torch.FloatTensor(mean).to(hsq.device)
        std = torch.FloatTensor(std).to(hsq.device)

        return mean, std

    def _get_hsq_score(x, q):
        x = x.select(-1, q - 1)
        if q in [1, 2, 20, 22, 34, 36]:
            return 100 - (x - 1) * 25
        elif q in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            return 50 * (x - 1)
        elif q in [13, 14, 15, 16, 17, 18, 19]:
            return 100 * (x - 1)
        elif q in [21, 23, 26, 27, 30]:
            return 100 - (x - 1) * 20
        elif q in [24, 25, 28, 29, 31]:
            return 20 * (x - 1)
        elif q in [32, 33, 35]:
            return 25 * (x - 1)

        raise ValueError

    hsq_mean, hsq_std = _get_stat(normalizers['hsq'])
    target_mean, target_std = _get_stat(normalizers['target'])

    hsq_recover = hsq * hsq_std + hsq_mean

    # batch * [surgical/non-surgical] * 8 years * 36
    hsq_recover = hsq_recover.view(-1, 2, nb_years, 36)

    hsq_bp = []
    for q in [21, 22]:
        hsq_bp.append(
            _get_hsq_score(hsq_recover, q).unsqueeze(dim=-1)
        )
    hsq_bp = torch.cat(hsq_bp, dim=-1)
    hsq_bp = torch.mean(hsq_bp, dim=-1, keepdim=True)

    hsq_pf = []
    for q in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        hsq_pf.append(
            _get_hsq_score(hsq_recover, q).unsqueeze(dim=-1)
        )
    hsq_pf = torch.cat(hsq_pf, dim=-1)
    hsq_pf = torch.mean(hsq_pf, dim=-1, keepdim=True)

    hsq_target = torch.cat([hsq_bp, hsq_pf], dim=-1)
    hsq_target = hsq_target.view(-1, 2 * nb_years * 2)

    # recover back with standard scaler
    hsq_target = (hsq_target - target_mean) / target_std

    return hsq_target


def run_one_epoch(args, model, ema_model, data_loader, eid, optimizer=None, normalizers=None):
    assert(normalizers is not None)

    running_loss = 0.0

    for bid, (feature, target, hsq, surgical) in enumerate(data_loader):
        if feature.shape[0]<batch_size:
            continue
        feature = feature.cuda()
        target = target.cuda().float()
        hsq = hsq.cuda().float()
        surgical = surgical.cuda().float()

        target_mask = torch.isnan(target)
        target = torch.nan_to_num(target)

        hsq_mask = torch.isnan(hsq)
        hsq = torch.nan_to_num(hsq)
        target_out, hsq_out, cls = model(feature)

        with torch.no_grad():
            ema_target_out, ema_hsq_out, _ = ema_model(feature)
            ema_target_hsq = calc_hsq(ema_hsq_out, normalizers)

        #loss_target_out = loss_func(target_out, target, reduction='none')
        #loss_target_out = _masked_loss(loss_target_out, target_mask)
        loss_target_out = loss_func(target_out, target, target_mask)

        target_hsq = calc_hsq(hsq_out, normalizers)
        #loss_target_hsq = loss_func(target_hsq, target, reduction='none')
        #loss_target_hsq = _masked_loss(loss_target_hsq, target_mask)
        loss_target_hsq = loss_func(target_hsq, target, target_mask)

        #loss_hsq_out = loss_func(hsq_out, hsq, reduction='none')
        #loss_hsq_out = _masked_loss(loss_hsq_out, hsq_mask)
        loss_hsq_out = loss_func(hsq_out, hsq, hsq_mask)

        cls_loss = F.binary_cross_entropy_with_logits(cls, surgical)

        # select_loss = _select_loss(model.selector)

        alpha = 0.5
        beta = 0.25
        lambda_ = 0.01
        # weight for ema-consistency
        theta = 0.03

        # alpha = 0.5
        # beta = 0.25
        # lambda_ = 0.05
        # # weight for ema-consistency
        # theta = 0.05
        
        loss = loss_target_out
        ema_loss = loss_func(ema_target_out, target_out, target_mask) + \
            loss_func(ema_target_hsq, target_out, target_mask) * alpha

        # only work for DL
        if optimizer is not None and args.use_deep=='True':
            if args.m == 'True':
                loss += loss_target_hsq * alpha
                loss += loss_hsq_out * beta
            if args.c == 'True':
                loss += cls_loss * lambda_
            if args.s == 'True':
                loss += ema_loss * theta
        # loss += select_loss * 0.5
        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        decay = 0.95
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1-decay)

    return running_loss / len(data_loader)

def predict(model, data, normalizers):
    model.eval()

    data_loader = build_data_loader(
        data, shuffle=False
    )
    
    feature, target, hsq, target_out, hsq_out, target_hsq = [], [], [], [], [], []
    for bid, (feature_, target_, hsq_, surgical) in enumerate(data_loader):
        if feature_.shape[0]<batch_size:
            continue
        feature_ = feature_.cuda()
        target_out_, hsq_out_, cls_ = model(feature_)
        target_hsq_ = calc_hsq(hsq_out_, normalizers)
        feature.append(feature_)
        target.append(target_)
        target_out.append(target_out_)
        hsq.append(hsq_)
        hsq_out.append(hsq_out_)
        target_hsq.append(target_hsq_)

    feature = torch.cat(feature, dim=0)
    target = torch.cat(target, dim=0)
    target_out = torch.cat(target_out, dim = 0)
    hsq = torch.cat(hsq, dim=0)
    hsq_out = torch.cat(hsq_out, dim=0)
    target_hsq = torch.cat(target_hsq, dim=0)
    
    feature_df = pd.DataFrame(
        feature.data.cpu().numpy(), columns=data['feature'].columns
    )
    target_df = pd.DataFrame(
        target.data.cpu().numpy(), columns=data['target'].columns
    )
    target_out_df = pd.DataFrame(
        target_out.data.cpu().numpy(), columns=data['target'].columns
    )

    hsq_df = pd.DataFrame(
        hsq.data.cpu().numpy(), columns=data['hsq'].columns
    )
    hsq_out_df = pd.DataFrame(
        hsq_out.data.cpu().numpy(), columns=data['hsq'].columns
    )

    target_hsq_df = pd.DataFrame(
        target_hsq.data.cpu().numpy(), columns=data['target'].columns
    )
    return {
        'target_df': target_df,
        'target_out_df': target_out_df,
        'hsq_df': hsq_df,
        'hsq_out_df': hsq_out_df,
        'target_hsq_df': target_hsq_df,
    }


def fit(args, model, train_data, test_data, normalizers):
    train_loader = build_data_loader(train_data, shuffle=True)
    test_loader = build_data_loader(test_data, shuffle=False)

    model.cuda()

    # mean teacher
    ema_model = copy.deepcopy(model)

    best_val_loss = 1e30
    best_model = None
    best_model_dict = None

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    early_stop = 30
    # epochs = 1000
    epochs = 1000

    patience = early_stop

    for eid in range(epochs):
        model.train()
        train_loss = run_one_epoch(
            args, model, ema_model, train_loader, eid, optimizer, normalizers=normalizers
        )

        model.eval()
        test_loss = run_one_epoch(
            args, model, ema_model, test_loader, eid, normalizers=normalizers
        )

        # selector = torch.cat(list(model.selector), dim=-1)

        # np.save('selector_{}'.format(nb_feature), selector.data.cpu().numpy())
        # print(selector)

        print(f'Epoch {eid}, train loss {train_loss}, val loss {test_loss}')

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model_dict = copy.deepcopy(model.state_dict())
            patience = early_stop
            # pickle.dump(model, open(f'./{args.use_deep}_{args.c}_{args.m}_{args.s}_best_model.pkl', 'wb'))
        else:
            patience = patience - 1

        if patience == 0:
            break
        
    # best_model = pickle.load(open(f'./{args.use_deep}_{args.c}_{args.m}_{args.s}_best_model.pkl', 'rb'))
    model.load_state_dict(best_model_dict)
    
    
    # return best_model
    return model

def _recover(result, normalizers):
    for key, df in result.items():
        if key == 'normalizers':
            continue
        print('Recovering {} ...'.format(key))
        norm_ = normalizers[key.split('_')[0]]
        for i, col in enumerate(df.columns):
            if isinstance(norm_[i], StandardScaler):
                df[col] = norm_[i].inverse_transform(
                    df[col].values.reshape(-1, 1)
                )
            else:
                df[col] = norm_[i].inverse_transform(
                    df[col].values.reshape(-1, )
                )

    return result


def show_metric(args, results, eps=10):
    def _bp_round(x):
        diff = np.abs(x - feasible_bp)
        return feasible_bp[np.argmin(diff)]

    def _pf_round(x):
        diff = np.abs(x - feasible_pf)
        return feasible_pf[np.argmin(diff)]

    def _nmae(pred, label):
        a = np.nanmean(
            np.abs(label - pred), axis=0
        )
        b = np.nanmean(
            np.abs(label), axis=0
        )
        return a / b

    def _mae(pred, label):
        loss = np.abs(pred - label)
        return np.nanmean(loss, axis=0)
    
    def _nrmse(pred, label):
        pred - label
        return np.sqrt(np.nanmean(np.square(pred - label)))/np.sqrt(np.nanmean(np.square(label))+1e-2)

    def func(x):
        return x.iloc[:8].corr(x.iloc[-8:])
    
    def func1(x):
        return x.iloc[-8:].count().sum()>4

    def corr(pred, label):
        n_col = [x for x in pred.columns if x.find('non') != -1]
        s_col = [x for x in pred.columns if x.find('non') == -1]
        p_n = pred[n_col]
        p_s = pred[s_col]
        l_n = label[n_col]
        l_s = label[s_col]

        rs_n = pd.concat([p_n, l_n], axis=1)
        rs1_n = rs_n.apply(func1, axis=1)
        rs_n = rs_n.drop(index=rs1_n[rs1_n==False].index.to_list())
        rs_n = rs_n.apply(func, axis=1)
        rs_n = rs_n[rs_n>0.1]

        rs_s = pd.concat([p_s, l_s], axis=1)
        rs1_s = rs_s.apply(func1, axis=1)
        rs_s = rs_s.drop(index=rs1_s[rs1_s==False].index.to_list())
        rs_s = rs_s.apply(func, axis=1)
        rs_s = rs_s[rs_s>0.1]

        corr_n = np.array(rs_n.values,dtype = float)
        corr_s = np.array(rs_s.values,dtype = float)
        return np.nanmean(np.concatenate([corr_n, corr_s]))

    bp_round_func = np.frompyfunc(_bp_round, 1, 1)
    pf_round_func = np.frompyfunc(_pf_round, 1, 1)

    bp_loss, pf_loss = {'nmae': [], 'mae': [], 'nrmse': [], 'corr': []}, {'nmae': [], 'mae': [], 'nrmse': [], 'corr': []}

    for result in results:
        if args.use_deep=='True' and args.m=='True':
            pred = (result['target_out_df'] + result['target_hsq_df']) * 0.5
        else:
            pred = result['target_out_df']
        label = result['target_df']

        bp_cols = [x for x in label.columns if x.find('_bp_') != -1]
        pf_cols = [x for x in label.columns if x.find('_pf_') != -1]

        bp_pred = bp_round_func(pred[bp_cols]).astype('float')
        bp_loss['nmae'].append(_nmae(bp_pred, label[bp_cols]))
        bp_loss['mae'].append(_mae(bp_pred, label[bp_cols]))
        bp_loss['nrmse'].append(_nrmse(bp_pred, label[bp_cols]))
        bp_loss['corr'].append(corr(bp_pred, label[bp_cols]))
        #pickle.dump(bp_pred, open('bp_pred.pkl','wb'))
        #pickle.dump(label[bp_cols], open('label[bp_cols].pkl','wb'))

        pf_pred = pf_round_func(pred[pf_cols]).astype('float')
        pf_loss['nmae'].append(_nmae(pf_pred, label[pf_cols]))
        pf_loss['mae'].append(_mae(pf_pred, label[pf_cols]))
        pf_loss['nrmse'].append(_nrmse(pf_pred, label[pf_cols]))
        pf_loss['corr'].append(corr(pf_pred, label[pf_cols]))

    mean_result = dict()
    std_result = dict()
    for key in ['nmae', 'mae', 'nrmse', 'corr']:
        #if key == 'nmae':
        #    axis = None
        #else:
        #    axis = 0
        axis = None

        print(
            'Overall BP {}: {} +/- {}'.format(
                key, np.mean(bp_loss[key], axis=axis), np.std(
                    bp_loss[key], axis=axis)
            )
        )
        print(
            'Overall PF {}: {} +/- {}'.format(
                key, np.mean(pf_loss[key], axis=axis), np.std(
                    pf_loss[key], axis=axis)
            )
        )
        mean_result[f'BP_{key}'] = np.mean(bp_loss[key], axis=axis)
        mean_result[f'PF_{key}'] = np.mean(pf_loss[key], axis=axis)
        std_result[f'PF_{key}'] = np.std(bp_loss[key], axis=axis)
        std_result[f'PF_{key}'] = np.std(pf_loss[key], axis=axis)
        final_result = {'mean':mean_result, 'std':std_result}
        np.save(open(f'./ablation_result/{args.use_deep}_{args.c}_{args.m}_{args.s}_{args.feature}.npy','wb'), final_result)


def add_noise_to_label(table, mean=0, std=0.3):
    values = table.values
    noisy = np.random.normal(mean, std, size=values.shape)
    not_none = ~np.isnan(values)
    values[not_none] = values[not_none] + noisy[not_none]

def add_noise_to_input(table, table_1, normalizers):
    index_n = table_1.values.reshape(-1) == 0
    index_s = table_1.values.reshape(-1) == 1
    norm_ = normalizers['feature']

    l = []
    for i, col in enumerate(table.columns):
        if isinstance(norm_[i], StandardScaler):
            l.append(col)
    values = table[l].values

    values_n = values[index_n].copy()
    noisy_n = np.random.normal(0.7, 0.1, size=values_n.shape)
    not_none_n = ~np.isnan(values_n)
    values_n[not_none_n] = values_n[not_none_n] + noisy_n[not_none_n]

    values_s = values[index_s].copy()
    noisy_s = np.random.normal(0, 0.1, values_s.shape)
    not_none_s = ~np.isnan(values_s)
    values_s[not_none_s] = values_s[not_none_s] + noisy_s[not_none_s]

    table.loc[:,l] = np.concatenate([values_n, values_s], axis=0)


def replace_with_nan(row, ratio=0.3):
    non_null_count = row.count()
    num_to_replace = int(non_null_count * ratio)
    replace_indices = np.random.choice(row.dropna().index, size=num_to_replace, replace=False)
    row[replace_indices] = np.nan
    return row


def run(args):
    data, obs_ID, rct_ID = load_raw_data()

    results = []
    models = []

    ID = np.concatenate([obs_ID, rct_ID], axis=0)
    kfold = KFold(n_splits=3)

    i=0
    for train_idx, test_idx in kfold.split(ID):
        train_ID, test_ID = ID[train_idx], ID[test_idx]
        
        if os.path.exists('train_feature_{}_{}.pkl'.format(i, args.feature)):
            train_feature = pickle.load(open('train_feature_{}_{}.pkl'.format(i, args.feature),'rb'))
            test_feature = pickle.load(open('test_feature_{}_{}.pkl'.format(i, args.feature),'rb'))
            normalizers = pickle.load(open('normalizers_{}_{}.pkl'.format(i, args.feature),'rb'))
        else:
            train_feature, test_feature, normalizers = load_feature(
                data.loc[train_ID].copy(),
                data.loc[test_ID].copy(), 
                args
            )
            pickle.dump(train_feature, open('train_feature_{}_{}.pkl'.format(i, args.feature),'wb'))
            pickle.dump(test_feature, open('test_feature_{}_{}.pkl'.format(i, args.feature),'wb'))
            pickle.dump(normalizers, open('normalizers_{}_{}.pkl'.format(i, args.feature),'wb'))

        # add_noise_to_label(train_feature['target'])
        # add_noise_to_label(test_feature['target'])

        # add_noise_to_input(train_feature['feature'], train_feature['surgical'], normalizers)
        # add_noise_to_input(test_feature['feature'], test_feature['surgical'], normalizers)

        # train_feature['target'] = train_feature['target'].apply(replace_with_nan, axis=1) 
        # test_feature['target'] = test_feature['target'].apply(replace_with_nan, axis=1) 

        i = i + 1
        model = Model(normalizers['feature'])

        model = fit(
            args, model, train_feature, test_feature, normalizers=normalizers
        )
        result = predict(model, test_feature, normalizers=normalizers)
        # result['feature'] = test_feature['feature']
        result = _recover(result, normalizers)

        results.append(result)
        # models.append(model.state_dict())
        models.append(model)

        break
        
    show_metric(args, results)

    #torch.save(results, './result/nn_res_transformer.cpt')
    #torch.save(models, './result/nn_model_transformer.cpt')
    torch.save(model.state_dict(), f'./result/transformer_model.pth')
    pickle.dump(results, open(f'./result/{args.use_deep}_{args.c}_{args.m}_{args.s}_{args.feature}_results_transformer_harv.pkl', 'wb'))
    pickle.dump(models, open(f'./result/{args.use_deep}_{args.c}_{args.m}_{args.s}_{args.feature}_models_transformer_harv.pkl', 'wb'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_deep', type=str, default='True')
    parser.add_argument('--c', type=str, default='True')
    parser.add_argument('--m', type=str, default='True')
    parser.add_argument('--s', type=str, default='True')
    parser.add_argument('--feature', type=str, default='20')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    run(args)
