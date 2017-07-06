%% This is the main program to classify Leukemia patients using SVM
clear;
clc;
close all;

%% Data preparation
load('golubsmall.mat');
% normTraindata = normc(traindata');
traindata = traindata';
trainclasses = trainclasses';
% normTestdata = normc(testdata');
testdata = testdata';
testclasses = testclasses';

% Choose 14 class I patients and 6 class II patients for training
% Choose 13 class I patients and 5 class II patients for validation
newTraindata(1:14, :) = traindata(1:14, :);
newTraindata(15:20, :) = traindata(28:33, :);
newTrainclasses(1:14) = trainclasses(1:14);
newTrainclasses(15:20) = trainclasses(28:33);
newTrainclasses = newTrainclasses';
newValidatedata(1:13, :) = traindata(15:27, :);
newValidatedata(14:18, :) = traindata(34:38, :);
newValidateclasses(1:13) = trainclasses(15:27);
newValidateclasses(14:18) = trainclasses(34:38);
newValidateclasses = newValidateclasses';

%% Linear, Polynomial, and RBF kernel SVM before PCA application
% Linear kernel
[nonpca_linCost, nonpca_linValAcc, nonpca_linValConf, nonpca_linTestAcc, ...
    nonpca_linTestConf] = doLinearSVM(newTraindata, newTrainclasses, ...
                                   newValidatedata, newValidateclasses, ...
                                   testdata, testclasses);

% Polynomial kernel
[nonpca_polyDegree, nonpca_polyValAcc, nonpca_polyValConf, nonpca_polyTestAcc, ...
    nonpca_polyTestConf] = doPolySVM(newTraindata, newTrainclasses, ...
                                   newValidatedata, newValidateclasses, ...
                                   testdata, testclasses);

% RBF kernel
[nonpca_rbfGamma, nonpca_rbfValAcc, nonpca_rbfValConf, nonpca_rbfTestAcc, ...
    nonpca_rbfTestConf] = doRbfSVM(newTraindata, newTrainclasses, ...
                                   newValidatedata, newValidateclasses, ...
                                   testdata, testclasses);

%% Linear, Polynomial, and RBF kernel SVM after PCA application
% PCA
[coeff, score, latent] = pca(traindata);
varianceCoverage = cumsum(latent) ./ sum(latent);
% Find the number of dimensions that covers 95% of the total variance
dimensionThreshold = find(varianceCoverage >= 0.95);
numDimension = dimensionThreshold(1);
newTraindata = newTraindata * coeff(:, 1:numDimension);
newValidatedata = newValidatedata * coeff(:, 1:numDimension);
newTestdata = testdata * coeff(:, 1:numDimension);

% Linear kernel
[pca_linCost, pca_linValAcc, pca_linValConf, pca_linTestAcc, ...
    pca_linTestConf] = doLinearSVM(newTraindata, newTrainclasses, ...
                                   newValidatedata, newValidateclasses, ...
                                   newTestdata, testclasses);

% Polynomial kernel
[pca_polyDegree, pca_polyValAcc, pca_polyValConf, pca_polyTestAcc, ...
    pca_polyTestConf] = doPolySVM(newTraindata, newTrainclasses, ...
                                   newValidatedata, newValidateclasses, ...
                                   newTestdata, testclasses);

% RBF kernel
[pca_rbfGamma, pca_rbfValAcc, pca_rbfValConf, pca_rbfTestAcc, ...
    pca_rbfTestConf] = doRbfSVM(newTraindata, newTrainclasses, ...
                                   newValidatedata, newValidateclasses, ...
                                   newTestdata, testclasses);

%% Summary
fprintf('The dimension has been reduced to %d\n', numDimension);
disp('########## Before PCA ##########');
disp('Linear SVM:');
fprintf('Cost = %f\n', nonpca_linCost);
fprintf('Validation Accuracy = %f\n', nonpca_linValAcc);
fprintf('Testing Accuracy = %f\n', nonpca_linTestAcc);

disp('Polynomial SVM:');
fprintf('Degree = %f\n', nonpca_polyDegree);
fprintf('Validation Accuracy = %f\n', nonpca_polyValAcc);
fprintf('Testing Accuracy = %f\n', nonpca_polyTestAcc);

disp('RBF SVM:');
fprintf('Gamma = %d\n', nonpca_rbfGamma);
fprintf('Validation Accuracy = %f\n', nonpca_rbfValAcc);
fprintf('Testing Accuracy = %f\n', nonpca_rbfTestAcc);

disp('########## After PCA ##########');
disp('Linear SVM:');
fprintf('Cost = %f\n', pca_linCost);
fprintf('Validation Accuracy = %f\n', pca_linValAcc);
fprintf('Testing Accuracy = %f\n', pca_linTestAcc);

disp('Polynomial SVM:');
fprintf('Degree = %f\n', pca_polyDegree);
fprintf('Validation Accuracy = %f\n', pca_polyValAcc);
fprintf('Testing Accuracy = %f\n', pca_polyTestAcc);

disp('RBF SVM:');
fprintf('Gamma = %d\n', pca_rbfGamma);
fprintf('Validation Accuracy = %f\n', pca_rbfValAcc);
fprintf('Testing Accuracy = %f\n', pca_rbfTestAcc);
