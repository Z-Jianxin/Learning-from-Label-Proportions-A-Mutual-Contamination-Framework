function [result_alter] = alterPSVM_cv(data_path)
load(data_path);
%clear all;
%load germanbank_cv.mat
num_folds = folds.k_fold;
Kernels = ["rbf1", "rbf01", "rbf001"];
%Dataset Name Format: german_bank_BagSize_[2]_Trial_[0].mat

%% kernel alter-pSVM with anealing
alter_p_Cs = [0.1, 1, 10];
alter_p_Cps = [1, 10, 100];
[temp1, temp2, temp3] = meshgrid(alter_p_Cs, alter_p_Cps, Kernels);
alter_p_params = [temp1(:), temp2(:), temp3(:)];
alter_p_cv_acc = zeros(1, size(alter_p_params, 1));
alter_p_cv_bacc = zeros(1, size(alter_p_params, 1));
alter_p_cv_bag_error = zeros(1, size(alter_p_params, 1));
para.method = 'alter-pSVM';
para.ep = 0;
N_random_cv = 1;
N_random = 1;
folds = eval("folds");
parfor i = 1:size(alter_p_params, 1)
    para_copy = para;
    para_copy.C = str2double(alter_p_params(i, 1));  % empirical loss weight
    para_copy.C_2 = str2double(alter_p_params(i, 2)); % proportion term weight
    cv_kernel_type = convertStringsToChars(alter_p_params(i, 3));
    avg_acc = 0.0;
    avg_bacc = 0.0;
    avg_bag_error = 0.0;
    %result_invcal = test_all_method(folds.fold_0.split, trK ,teK, para);
    for fold_j = 1:num_folds
        fold_id = int2str(fold_j - 1);
        fold_name = strcat('fold_', fold_id);
        %cv_fold = eval(fold_name);
        cv_fold = folds.(fold_name);
        K = kernel_f(cv_fold.data, cv_kernel_type);
        trK = K(cv_fold.split.train_data_idx, cv_fold.split.train_data_idx);
        teK = K(cv_fold.split.test_data_idx, cv_fold.split.train_data_idx);
        %result_invcal = test_all_method(cv_fold.split, trK ,teK, para);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result = [];
        obj = zeros(N_random_cv,1);
        for pp = 1:N_random_cv
            para_copy.init_y = ones(length(trK),1);
            r = randperm(length(trK));
            para_copy.init_y(r(1:floor(length(trK)/2))) = -1;
            result{pp} = test_all_method(cv_fold.split, trK, teK, para_copy);
            if isfield(result{pp}.model, 'obj')
                obj(pp) = result{pp}.model.obj;
            else
                obj(pp) = Inf;
            end
        end
        [mm,id] = min(obj);
        result_alter = result{id};
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        avg_acc = avg_acc + result_alter.test_acc/double(num_folds);
        avg_bacc = avg_bacc + result_alter.test_bacc/double(num_folds);
        avg_bag_error = avg_bag_error + result_alter.test_bag_error/double(num_folds);
    end
    alter_p_cv_acc(i) = avg_acc;
    alter_p_cv_bacc(i) = avg_bacc;
    alter_p_cv_bag_error(i) = avg_bag_error;
end
[max_value, max_idx] = min(alter_p_cv_bag_error);
para.C = str2double(alter_p_params(max_idx, 1));
para.C_2 = str2double(alter_p_params(max_idx, 2));
best_kernel_type = convertStringsToChars(alter_p_params(max_idx, 3));
K = kernel_f(data, best_kernel_type);
trK = K(split.train_data_idx, split.train_data_idx);
teK = K(split.test_data_idx, split.train_data_idx);

% extra test scheme
extra_teK.test_10 = K(split.test_10_data_idx, split.train_data_idx);
extra_teK.test_25 = K(split.test_25_data_idx, split.train_data_idx);
extra_teK.test_50 = K(split.test_50_data_idx, split.train_data_idx);
extra_teK.test_75 = K(split.test_75_data_idx, split.train_data_idx);
extra_teK.test_90 = K(split.test_90_data_idx, split.train_data_idx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result = [];
        obj = zeros(N_random,1);
        for pp = 1:N_random
            para.init_y = ones(length(trK),1);
            r = randperm(length(trK));
            para.init_y(r(1:floor(length(trK)/2))) = -1;
            result{pp} = test_all_method(split, trK, teK, para, extra_teK);
            if isfield(result{pp}.model, 'obj')
                obj(pp) = result{pp}.model.obj;
            else
                obj(pp) = Inf;
            end
            %obj(pp) = result{pp}.model.obj;
        end
        [mm,id] = min(obj);
        result_alter = result{id};
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%alter_p_params
%alter_p_cv_acc
%alter_p_cv_bacc
%alter_p_cv_bag_error
%para
%best_kernel_type
%result_alter
end