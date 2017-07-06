function [bestCost, validateAccuracy, validateConfMat, ...
          testAccuracy, testConfMat] = doLinearSVM(trainData, trainClasses, ...
                                            validateData, validateClasses, ...
                                            testData, testClasses)
% function: This function does SVM classification using Linear kernel function
%           It looks for the value of cost that produces the best accuracy
%           before using the best cost value to test

%% Grid search - Look for the value of cost that produces the best accuracy
cost = linspace(0.1, 10, 100);
validationAccuracies = zeros(size(cost));
for i = 1:size(cost, 2)
    svmOption = ['-q -s 0 -t 0 -c ', num2str(cost(i))];
    mdl = svmtrain(trainClasses, trainData, svmOption);
    labels = svmpredict(validateClasses, validateData, mdl);
    confMat = confusionmat(validateClasses, labels);
    validationAccuracies(i) = sum(diag(confMat)) / sum(confMat(:));
end

%% Visualize cost vs. accuracies
figure;
plot(cost, validationAccuracies, '.-');
title('Linear SVM - Determining Cost Value');
xlabel('Cost');
ylabel('Validation Accuracy');
% Get the best cost
[~, idx] = max(validationAccuracies);
bestCost = cost(idx);

%% Use the calculated cost value to train and validate again in order to test
svmOption = ['-q -s 0 -t 0 -c ', num2str(bestCost)];
mdl = svmtrain(trainClasses, trainData, svmOption);
validateLabel = svmpredict(validateClasses, validateData, mdl);
validateConfMat = confusionmat(validateClasses, validateLabel);
validateAccuracy = sum(diag(validateConfMat)) / sum(validateConfMat(:));
testLabel = svmpredict(testClasses, testData, mdl);
testConfMat = confusionmat(testClasses, testLabel);
testAccuracy = sum(diag(testConfMat)) / sum(testConfMat(:));

end % function
