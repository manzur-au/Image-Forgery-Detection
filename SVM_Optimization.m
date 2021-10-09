% Author: Mohammad Manzurul Islam
% Code created: 20/11/2018
% Last Modified: 11/5/2019
% A Program for optimizing SVM - a part of code for implementing the paper:
% A Robust Forgery Detection Method for Copy–Move and Splicing Attacks in Images
% https://doi.org/10.3390/electronics9091500

tic;


%% Uncomment this section for combined blocks
% blocksize = '4x8x16';
% dataset = 'FBDDF_Proposed_tif_uncompressed_scaling_115_110_105_1_95_90_85_Gray_4x8x16';
% featureset = 'FBDDF_Proposed_tif_uncompressed_scaling_115_110_105_1_95_90_85_Gray_4x8x16.mat';

%% For different block sizes

% Put a for loop for 'bzises'. In the paper, we used 4, 8 and 16 block size.

blocksize = bsizes;
dataset = 'NAME OF YOUR FEATURESET FILE WITHOUT FILE EXTENSION';
featureset = sprintf('%s_%dx%d.mat',dataset,blocksize,blocksize);

S = load(featureset);
X_ = S.data(:,1:end-1);
Y_ = S.data(:,end);
N_ = length(Y_);

rng(1);
seeds = randperm(100000,1000);


kfold = 10;
evals = 100;
standardize = false;
reducedtrain = false;
NN = 6000;
NN = min(NN,N_);

rpt = 20;
rpt = rpt + 1 - mod(rpt,2);
verbose = 1;

models_ = [];
for r = 1:rpt
    r
    rng(seeds(r));
    cvp_ = cvpartition(Y_,'KFold',kfold,'Stratify',true);
    
    if reducedtrain
        cvpH = cvpartition(Y_,'HoldOut',N_ - NN,'Stratify',true);
        X = X_(cvpH.training,:);
        Y = Y_(cvpH.training);
        N = length(Y);
        cvp = cvpartition(Y,'KFold',kfold,'Stratify',true);
    else
        X = X_;
        Y = Y_;
        N = N_;
        cvp = cvp_;
    end
    
    hoo = struct('Optimizer','bayesopt','ShowPlots',false,'CVPartition',cvp,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',evals,'Verbose',verbose);
    
    SVMModel = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',standardize,...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',hoo);
    
    CVModel = fitcsvm(X_,Y_,'KernelFunction','rbf','Standardize',false,...
        'CrossVal','on','CVPartition',cvp_,...
        'BoxConstraint',SVMModel.ModelParameters.BoxConstraint,...
        'KernelScale',SVMModel.ModelParameters.KernelScale);
    
    label = nan(N_,1);
    for i = 1:kfold
        label(cvp_.test(i)) = predict(CVModel.Trained{i},X_(cvp_.test(i),:));
    end
    
    [~,cm] = confusion(double(Y_' == [0;1]),double(label' == [0;1]));
    
    TN = cm(1,1);FP = cm(1,2);FN = cm(2,1);TP = cm(2,2);
    accuracy = ((TP+TN)/N_)*100;
    FNR = FN/(FN+TP)*100;
    FPR = FP/(FP+TN)*100;
    specificity = TN/(TN+FP)*100;
    sensitivity = TP/(TP+FN)*100;
    
    data = struct(...
        'RepeatNo',r,...
        'CVPartition',cvp,...
        'SVMModel',SVMModel,...
        'CVModel',CVModel,...
        'BoxConstraint',SVMModel.ModelParameters.BoxConstraint,...
        'KernelScale',SVMModel.ModelParameters.KernelScale,...
        'ConfusionMatrix',cm,...
        'Accuracy',accuracy,...
        'specificity',specificity,...
        'sensitivity',sensitivity,...
        'FNR',FNR,...
        'FPR',FPR)
    models_ = [models_;data];
end

models = sortrows(struct2table(models_,'AsArray',true),{'Accuracy','FNR'},{'descend','ascend'});
result = struct(...
    'Seeds',seeds,...
    'Dataset',dataset,...
    'Blocksize',blocksize,...
    'Featureset',featureset,...
    'Standardize',standardize,...
    'KFold',kfold,...
    'Evals',evals,...
    'Models',models);


resultSVM = sprintf('%s_resultSVM.mat',dataset);
save(resultSVM, 'result');

% Code for resultset with all models
median_acc = result.Models(11,:)
avg = mean(table2array(result.Models(1:20,8:10)))
max = result.Models(1,8:10)

toc;