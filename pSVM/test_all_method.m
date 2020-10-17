function result = test_all_method(split, trK, teK, para, extra_teK)

% this wrapper function can be used for both tuning parameters and training

% input1: train_data, train_bag_idx, train_bag_prop, test_label
% input2: test_data, test_bag_idx, test_bag_prop, test_label
% input3: parameters

% output1: model, predicted_train_label, train_acc, train_acc_bag
% output2: predicted_test_label, test_acc, test_acc_bag

% note that test_acc_bag (error) can be used for cross-validation with parameter
% tuning
 
switch (para.method)
%     case 'mean_map'
%         model = mean_map(trK, split.train_bag_idx, split.train_bag_prop, para);
%         if strcmp(para.mode,'kernel')
%             predict_response = sign(model.alpha'*teK' + model.b)';
%         else
%             predict_response = model.w'*teK' + model.b;
%             predict_response = predict_response';
%         end
%     case 'pSVM_linear'
%         model = alternating_svm_linear(trK, split.train_bag_idx, split.train_bag_prop, para);
%         predict_response = teK*model.w + model.b;
%     case 'pSVM_linear_anealing'
%         C = para.C;
%         para.C = 10e-5 * C;
%         while(para.C ~= C)
%             para.C = min(para.C*1.5,C);
%             model = alternating_svm_linear(trK, split.train_bag_idx, split.train_bag_prop, para);
%             para.init_y = model.y;
%         end
%         predict_response = teK*model.w + model.b;
%     case 'regularSVM_linear'
%         model = regular_svm_wrapper_linear(sparse(trK), split.train_label, para);
%         predict_response = teK*model.w + model.b;
%         model.y = trK*model.w + model.b;
%         model.y(model.y>0) = 1;
%         model.y(model.y<=0) = -1;
%     case 'pSVM'
%         model = alternating_svm_dual_with_slack_v5(trK, split.train_bag_idx, split.train_bag_prop, para);
%         predict_response =  teK(:, model.support_v) * model.alp + model.b;
%     case 'pSVM_no_slack'
%         model = alternating_svm_dual(trK, split.train_bag_idx, split.train_bag_prop, para);
%         predict_response =  teK(:, model.support_v) * model.alp + model.b;
    case 'InvCal'
        model = invcal_dual(trK, split.train_bag_idx, split.train_bag_prop, para);
        predict_response = invcal_dual_predict(teK, model);
%     case 'regularSVM'
%         model = regular_svm_wrapper(trK, split.train_label, para);
%         predict_response = teK(:, model.support_v) * model.alp + model.b;
%         model.y = trK(:, model.support_v) * model.alp + model.b;
%         model.y(model.y>0) = 1;
%         model.y(model.y<=0) = -1;
    case 'conv-pSVM'
        model = LGMMC_train(trK, split.train_bag_idx, split.train_bag_prop, para);
        predict_response = sign(teK*model.predict_model);
    case 'alter-pSVM'
        para.verbose = 0;
        C = para.C;
        para.C = 10e-5 * C;
        while(para.C ~= C)
            para.C = min(para.C*1.5,C);
            model = alternating_svm_dual_with_slack(trK, split.train_bag_idx, split.train_bag_prop, para);
            para.init_y = model.y;
        end
        if (isfield(model,'support_v'))
            predict_response =  teK(:, model.support_v) * model.alp + model.b;
        else
            predict_response = ones(size(split.test_label));
            model.obj = Inf;
        end
end

%% the the roc curve
if size(unique(split.test_label), 1) > 1
    [roc.roc_x, roc.roc_y, roc.thresholds, roc.auc, roc.opt, roc.suby, roc.subynames] = perfcurve(split.test_label, predict_response, 1);
    result.roc = roc;
    result.auc = roc.auc;
end
%% get the confusion matrix
%[cm.confusion_value, cm.confusion_matrix, cm.ind, cm.normalized_cm] = confusion(reshape(split.test_label(:) == 1, 1, []), ...
%                                                                                    reshape(predict_response(:) > 0, 1, []));
result.cm = confusionmat(split.test_label(:) == 1, predict_response(:) > 0);

%% get performance from predict_response
predict_label = predict_response;
predict_label(predict_label>0) = 1;
predict_label(predict_label<=0) = -1;

result.predicted_train_label = model.y;
result.predicted_test_label = predict_label;

result.train_acc = length(find(result.predicted_train_label - split.train_label == 0))/ length(model.y);
result.test_acc = length(find(result.predicted_test_label - split.test_label==0))/length(predict_label);

result.test_bacc = 0.5*sum((result.predicted_test_label(:) == 1).*(split.test_label(:)==1))/sum((split.test_label(:)==1)) + 0.5*sum((result.predicted_test_label(:) == -1).*(split.test_label(:)==-1))/sum((split(:).test_label==-1));

result.train_bag_error = get_bag_err(result.predicted_train_label, split.train_bag_idx, split.train_bag_prop);
result.test_bag_error = get_bag_err(result.predicted_test_label, split.test_bag_idx, split.test_bag_prop);

result.model = model;


if exist('extra_teK', 'var')
    switch(para.method)
        case 'InvCal'
            [acc_10, bacc_10] = test_invCal(extra_teK.test_10, result.model, split.test_10_label);
            [acc_25, bacc_25] = test_invCal(extra_teK.test_25, result.model, split.test_25_label);
            [acc_50, bacc_50] = test_invCal(extra_teK.test_50, result.model, split.test_50_label);
            [acc_75, bacc_75] = test_invCal(extra_teK.test_75, result.model, split.test_75_label);
            [acc_90, bacc_90] = test_invCal(extra_teK.test_90, result.model, split.test_90_label);
        case 'alter-pSVM'
            [acc_10, bacc_10] = test_pSVM(extra_teK.test_10, result.model, split.test_10_label);
            [acc_25, bacc_25] = test_pSVM(extra_teK.test_25, result.model, split.test_25_label);
            [acc_50, bacc_50] = test_pSVM(extra_teK.test_50, result.model, split.test_50_label);
            [acc_75, bacc_75] = test_pSVM(extra_teK.test_75, result.model, split.test_75_label);
            [acc_90, bacc_90] = test_pSVM(extra_teK.test_90, result.model, split.test_90_label);
    end
    
    result.acc_10 = acc_10;
    result.acc_25 = acc_25;
    result.acc_50 = acc_50;
    result.acc_75 = acc_75;
    result.acc_90 = acc_90;
    
    result.bacc_10 = bacc_10;
    result.bacc_25 = bacc_25;
    result.bacc_50 = bacc_50;
    result.bacc_75 = bacc_75;
    result.bacc_90 = bacc_90;
end


%% this is only for convexpSVM
% reverse the labels
if strcmp(para.method, 'convexpSVM')
    result_2.predicted_test_label = -result.predicted_test_label;
    result_2.predicted_train_label = -result.predicted_train_label;
    result_2.train_acc = 1-result.train_acc;
    result_2.test_acc = 1-result.test_acc;
    result_2.train_bag_error = get_bag_err(result_2.predicted_train_label, split.train_bag_idx, split.train_bag_prop);
    result_2.test_bag_error = get_bag_err(result_2.predicted_test_label, split.test_bag_idx, split.test_bag_prop);
    result_2.model = result.model;
    result_2.model.y = -result.model.y;
    %result_2.model.y_al = -result.model.y_al;
    result_2.model.predict_model = -result.model.predict_model;
    result_2.model.if_reversed = 1;
    if result_2.train_acc > result.train_acc
        result = result_2;
    end
end

end

function err = get_bag_err(predicted_label, bag_idx, bag_prop)
err = 0;
for i = 1:length(bag_prop) % bag_idx
    p_label = predicted_label(bag_idx == i);
    p_prop = length(find(p_label==1))/length(p_label);
    err = err + abs(p_prop - bag_prop(i))* length(p_label)/length(predicted_label);
end
end

function [test_acc, test_bacc] = test_pSVM(teK, model, test_label)
    score = teK(:, model.support_v) * model.alp + model.b;
    pred = score;
    pred(score>0) = 1;
    pred(score<=0) = -1;
    test_acc = length(find(pred - test_label==0))/length(score);
    test_bacc = 0.5*sum((pred(:) == 1).*(test_label(:)==1))/sum(test_label(:)==1) + 0.5*sum((pred(:) == -1).*(test_label(:)==-1))/sum(test_label(:)==-1);
end

function [test_acc, test_bacc] = test_invCal(teK, model, test_label)
    score = invcal_dual_predict(teK, model); %length(find(result.predicted_test_label - split.test_label==0))/length(predict_label);
    pred = score;
    pred(score>0) = 1;
    pred(score<=0) = -1;
    test_acc = length(find(pred - test_label==0))/length(score);
    test_bacc = 0.5*sum((pred(:) == 1).*(test_label(:)==1))/sum(test_label(:)==1) + 0.5*sum((pred(:) == -1).*(test_label(:)==-1))/sum(test_label(:)==-1);
end