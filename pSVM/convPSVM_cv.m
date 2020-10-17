function [result_conv_p] = convPSVM_cv(data_path)
%clear all;
%load germanbank_cv_new.mat
load(data_path);
num_folds = folds.k_fold;
Kernels = ["rbf1", "rbf01", "rbf001"];
%% kernel conv-pSVM
% with SVM form as in Li et al. Tigher and convex maximum margin clustering
conv_p_Cs = [0.1, 1, 10];
conv_p_eps = [0, 0.01, 0.1];
[temp1, temp2, temp3] = meshgrid(conv_p_Cs, conv_p_eps, Kernels);
conv_p_params = [temp1(:), temp2(:), temp3(:)];
conv_p_cv_acc = zeros(1, size(conv_p_params, 1));
conv_p_cv_bacc = zeros(1, size(conv_p_params, 1));
conv_p_cv_bag_error = zeros(1, size(conv_p_params, 1));
para.method = 'conv-pSVM';

for i = 1:size(conv_p_params, 1)
    para.C = str2double(conv_p_params(i, 1));
    para.ep = str2double(conv_p_params(i, 2));
    cv_kernel_type = convertStringsToChars(conv_p_params(i, 3));
    avg_acc = 0.0;
    avg_bacc = 0.0;
    avg_bag_error = 0.0;
    for fold_j = 1:num_folds
        fold_id = int2str(fold_j - 1);
        fold_name = strcat('folds.fold_', fold_id);
        cv_fold = eval(fold_name);
        K = kernel_f(cv_fold.data, cv_kernel_type);
        trK = K(cv_fold.split.train_data_idx, cv_fold.split.train_data_idx);
        teK = K(cv_fold.split.test_data_idx, cv_fold.split.train_data_idx);
        % set the max number of iterations
        para.max_iteration = 5;
        %
        result_conv_p = test_all_method(cv_fold.split, trK ,teK, para);
        avg_acc = avg_acc + result_conv_p.test_acc/double(num_folds);
        avg_bacc = avg_bacc + result_conv_p.test_bacc/double(num_folds);
        avg_bag_error = avg_bag_error + result_conv_p.test_bag_error/double(num_folds);
    end
    conv_p_cv_acc(i) = avg_acc;
    conv_p_cv_bacc(i) = avg_bacc;
    conv_p_cv_bag_error(i) = avg_bag_error;
end
[max_value, max_idx] = min(conv_p_cv_bag_error);
para.C = str2double(conv_p_params(max_idx, 1));
para.ep = str2double(conv_p_params(max_idx, 2));
best_kernel_type = convertStringsToChars(conv_p_params(max_idx, 3));
K = kernel_f(data, best_kernel_type);
trK = K(split.train_data_idx, split.train_data_idx);
teK = K(split.test_data_idx, split.train_data_idx);
result_conv_p = test_all_method(split, trK ,teK, para);

%TODO: save these to files 
%conv_p_params
%conv_p_cv_acc
%conv_p_cv_bacc
%para
%best_kernel_type
%result_conv_p
%return;
end
