% BIO600, SDSU
% Author: Xin Zhou

clear;
close all;
load('golubsmall','traindata','trainclasses','testdata','testclasses');

% *********************** preprocessing *********************** %
traindata = double(traindata);
trainclasses  = double(trainclasses);

testdata = double(testdata);
testclasses  = double(testclasses);

selects = 2;
covarience_matrix = traindata*traindata';
[COEFF,SCORE,LATENT] = princomp(traindata');
% [u s v] = svd(covarience_matrix);
% features = u(:,1:selects);
features = COEFF(:,1:selects);
train_inputs = features'*traindata;

covarience_matrix_t = testdata*testdata';
[COEFF2,SCORE,LATENT] = princomp(testdata');
% [u_t s_t v_t] = svd(covarience_matrix_t);
% features_t = u_t(:,1:selects);
features_t = COEFF2(:,1:selects);
test_inputs = features_t'*testdata;

% ************************ Normalize ************************* %
for k =1:selects
    train_inputs(k,:) = (train_inputs(k,:) - min(train_inputs(k,:)))./max(train_inputs(k,:));
    test_inputs(k,:) = (test_inputs(k,:) - min(test_inputs(k,:)))./max(test_inputs(k,:));
end
% train_inputs = normc(train_inputs')';
% test_inputs = normc(test_inputs')';

x1 = [];
y1 = [];
x2 = [];
y2 = [];
for k = 1:38
    if trainclasses(k) == 1
        x1 = [x1 train_inputs(1,k)];
        y1 = [y1 train_inputs(2,k)];
    else
        x2 = [x2 train_inputs(1,k)];
        y2 = [y2 train_inputs(2,k)];
    end
end

scatter(x1, y1, 'x', 'r');
hold on;
scatter(x2, y2, 'o', 'b');



% ******************* Training  ******************* %
sum_square_errors = 20;
while(sum_square_errors > 4)
    net = feedforwardnet(10);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;


%     indexes = randperm(38);
%     train_inputs = train_inputs(:,indexes);
%     trainclasses = trainclasses(:, indexes);

    net = train(net,train_inputs,trainclasses);

    threshold = 1.5;

    outputs = net(train_inputs);
    sum_square_errors = trainclasses - outputs;
    sum_square_errors = sum(sum_square_errors.*sum_square_errors)

    L = length(outputs);
    new_outputs = zeros(1, L);
    for k = 1:length(outputs)
        if outputs(k) < threshold
            new_outputs(k) = 1;
        else
            new_outputs(k) = 2;
        end
    end

    accuracy =  (L - sum(abs(trainclasses-new_outputs)))/L;
    fprintf('train accuracy %f \n',accuracy);

    
end

test_outputs = net(test_inputs);
sum_square_errors = testclasses - test_outputs;
sum_square_errors = sum(sum_square_errors.*sum_square_errors)

L = length(test_outputs);
new_outputs = zeros(1,L);
for k = 1:length(test_outputs)
    if test_outputs(k) < threshold
        new_outputs(k) = 1;
    else
        new_outputs(k) = 2;
    end
end
accuracy =  (L - sum(abs(testclasses-new_outputs)))/L;
fprintf('test accuracy %f \n',accuracy);
