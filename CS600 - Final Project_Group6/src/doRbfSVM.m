function [bestGamma, validateAccuracy, validateConfMat, ...
          testAccuracy, testConfMat] = doRbfSVM(trainData, trainClasses, ...
                                            validateData, validateClasses, ...
                                            testData, testClasses)
% function: This function does SVM classification using RBF kernel function
%           It looks for the value of gamma that produces the best accuracy
%           before using the best gamma value test

%% Grid search - Look for the gamma value that produces the best accuracy
% gamma = linspace(0, 200, 51);
gamma = linspace(1, 100, 100) * (10e-12);
validationAccuracies = zeros(size(gamma));
for i = 1:size(gamma, 2)
    svmOption = ['-q -s 0 -t 2 -g ', num2str(gamma(i))];
    mdl = svmtrain(trainClasses, trainData, svmOption);
    labels = svmpredict(validateClasses, validateData, mdl);
    confMat = confusionmat(validateClasses, labels);
    validationAccuracies(i) = sum(diag(confMat)) / sum(confMat(:));
end

%% Visualize gamma vs. accuracies
figure;
plot(gamma, validationAccuracies, '.-');
title('RBF - Determining Gamma Value');
xlabel('Gamma');
ylabel('Validation Accuracy');
% Get the best gamma
[~, idx] = max(validationAccuracies);
bestGamma = gamma(idx);

%% Use the calculated gamma value to train, validate, and test
svmOption = ['-q -s 0 -t 2 -g ', num2str(bestGamma)];
mdl = svmtrain(trainClasses, trainData, svmOption);
validateLabel = svmpredict(validateClasses, validateData, mdl);
validateConfMat = confusionmat(validateClasses, validateLabel);
validateAccuracy = sum(diag(validateConfMat)) / sum(validateConfMat(:));
testLabel = svmpredict(testClasses, testData, mdl);
testConfMat = confusionmat(testClasses, testLabel);
testAccuracy = sum(diag(testConfMat)) / sum(testConfMat(:));

end % function
