function [result_invcal] = InvCal_cv(data_path)
%clear all;
%init();
%load heart.mat;
%load germanbank_cv.mat
load(data_path);
%Dataset Name Format: german_bank_BagSize_[2]_Trial_[0].mat
num_folds = folds.k_fold;
Kernels = ["rbf1", "rbf01", "rbf001"];
%% kernel InvCal
InvCal_Cps = [0.1, 1, 10];
InvCal_eps = [0, 0.01, 0.1];
[temp1, temp2, temp3] = meshgrid(InvCal_Cps, InvCal_eps, Kernels);
InvCal_params = [temp1(:), temp2(:), temp3(:)];
Invcal_cv_acc = zeros(1, size(InvCal_params, 1));
Invcal_cv_bacc = zeros(1, size(InvCal_params, 1));
Invcal_cv_bag_error = zeros(1, size(InvCal_params, 1));
para.method = 'InvCal';
folds = eval("folds");
parfor i = 1:size(InvCal_params, 1)
    para_copy = para;
    para_copy.C = str2double(InvCal_params(i, 1));
    para_copy.ep = str2double(InvCal_params(i, 2));
    cv_kernel_type = convertStringsToChars(InvCal_params(i, 3));
    avg_acc = 0.0;
    avg_bacc = 0.0;
    avg_bag_error = 0.0;
    for fold_j = 1:num_folds
        fold_id = int2str(fold_j - 1);
        fold_name = strcat('fold_', fold_id);
        %cv_fold = eval(fold_name);
        cv_fold = folds.(fold_name);
        K = kernel_f(cv_fold.data, cv_kernel_type);
        trK = K(cv_fold.split.train_data_idx, cv_fold.split.train_data_idx);
        teK = K(cv_fold.split.test_data_idx, cv_fold.split.train_data_idx);
        result_invcal = test_all_method(cv_fold.split, trK ,teK, para_copy);
        avg_acc = avg_acc + result_invcal.test_acc/double(num_folds);
        avg_bacc = avg_bacc + result_invcal.test_bacc/double(num_folds);
        avg_bag_error = avg_bag_error + result_invcal.test_bag_error/double(num_folds);
    end
    Invcal_cv_acc(i) = avg_acc;
    Invcal_cv_bacc(i) = avg_bacc;
    Invcal_cv_bag_error(i) = avg_bag_error;
end
[max_value, max_idx] = min(Invcal_cv_bag_error);
para.C = str2double(InvCal_params(max_idx, 1));
para.ep = str2double(InvCal_params(max_idx, 2));
best_kernel_type = convertStringsToChars(InvCal_params(max_idx, 3));
K = kernel_f(data, best_kernel_type);
trK = K(split.train_data_idx, split.train_data_idx);
teK = K(split.test_data_idx, split.train_data_idx);

% extra test scheme
extra_teK.test_10 = K(split.test_10_data_idx, split.train_data_idx);
extra_teK.test_25 = K(split.test_25_data_idx, split.train_data_idx);
extra_teK.test_50 = K(split.test_50_data_idx, split.train_data_idx);
extra_teK.test_75 = K(split.test_75_data_idx, split.train_data_idx);
extra_teK.test_90 = K(split.test_90_data_idx, split.train_data_idx);

result_invcal = test_all_method(split, trK ,teK, para, extra_teK);
%TODO: save these to files 
%fileID = fopen(output_name,'w');
%fprintf(fileID,InvCal_params);
%fprintf(fileID,Invcal_cv_acc);
%fprintf(fileID,Invcal_cv_bacc);
%fprintf(fileID,Invcal_cv_bag_error);
%fprintf(fileID,para);
%fprintf(fileID,best_kernel_type);
%fprintf(fileID, 'acc=%.4f, bacc=%.4f, bag_error=%.4f\n', result_invcal.test_acc, result_invcal.test_bacc, result_invcal.test_bag_error);
%fclose(fileID);
end