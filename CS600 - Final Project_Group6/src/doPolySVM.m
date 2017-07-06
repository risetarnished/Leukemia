function [bestDegree, validateAccuracy, validateConfMat, ...
          testAccuracy, testConfMat] = doPolySVM(trainData, trainClasses, ...
                                            validateData, validateClasses, ...
                                            testData, testClasses)
% function: This function does SVM classification using polynomial kernel function
%           It looks for the value of degree that produces the best accuracy
%           before using the best degree value to test

%% Grid search - Look for the value of degree that produces the best accuracy
degree = linspace(0.1, 10, 100);
validationAccuracies = zeros(size(degree));
for i = 1:size(degree, 2)
    svmOption = ['-q -s 0 -t 1 -d ', num2str(degree(i))];
    mdl = svmtrain(trainClasses, trainData, svmOption);
    labels = svmpredict(validateClasses, validateData, mdl);
    confMat = confusionmat(validateClasses, labels);
    validationAccuracies(i) = sum(diag(confMat)) / sum(confMat(:));
end

%% Visualize degree vs. accuracies
figure;
plot(degree, validationAccuracies, '.-');
title('Polynomial - Determining Degree Value');
xlabel('Degree');
ylabel('Validation Accuracy');
% Get the best degree
[~, idx] = max(validationAccuracies);
bestDegree = degree(idx);

%% Use the calculated degree value to train and validate again in order to test
svmOption = ['-q -s 0 -t 1 -d ', num2str(bestDegree)];
mdl = svmtrain(trainClasses, trainData, svmOption);
validateLabel = svmpredict(validateClasses, validateData, mdl);
validateConfMat = confusionmat(validateClasses, validateLabel);
validateAccuracy = sum(diag(validateConfMat)) / sum(validateConfMat(:));
testLabel = svmpredict(testClasses, testData, mdl);
testConfMat = confusionmat(testClasses, testLabel);
testAccuracy = sum(diag(testConfMat)) / sum(testConfMat(:));

end % function
