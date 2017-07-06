clear;
clc;
close all;

load('golubsmall.mat')
testdata=testdata';
traindata=traindata';
trainclasses = trainclasses';
testclasses = testclasses';

%for i=1:2000
%   testmax=max(testdata(:,i));
%    testmin=min(testdata(:,i));
%    trainmax=max(traindata(:,i));
%    trainmin=min(traindata(:,i));
%    testdata(:,i)=(testdata(:,i)-testmin)/(testmax-testmin);
%    traindata(:,i)=(traindata(:,i)-trainmin)/(trainmax-trainmin);
%end

%%%%%%%%%%%%%%%%%%%%%choose different k
traindatachoose(1:14,:)=traindata(1:14,:);
traindatachoose(15:20,:)=traindata(28:33,:);
trainclasseschoose(1:14)=trainclasses(1:14);
trainclasseschoose(15:20)=trainclasses(28:33);
trainclasseschoose = trainclasseschoose';
validationdata(1:13,:)=traindata(15:27,:);
validationdata(14:18,:)=traindata(34:38,:);
validationclasses(1:13)=trainclasses(15:27);
validationclasses(14:18)=trainclasses(34:38);
validationclasses = validationclasses';

NumberNei=1:2:10;
Accuracy=[];
time=1:2:10;
for ii=1:5
    Knn=fitcknn(traindatachoose,trainclasseschoose,'NumNeighbors',NumberNei(ii));
    predictlabel=predict(Knn,validationdata);
    confmat = confusionmat(validationclasses, predictlabel);
    Accuracy(ii) = sum(diag(confmat)) / sum(confmat(:));
end

figure
plot(time, Accuracy)
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%
NumberNei=1:2:10;
Accuracy=[];
time=1:2:10;
for ii=1:5
    Knn=fitcknn(traindata,trainclasses,'NumNeighbors',NumberNei(ii));
    predictlabel=predict(Knn,testdata);
    confmat = confusionmat(testclasses, predictlabel);
    Accuracy(ii) = sum(diag(confmat)) / sum(confmat(:));
end

plot(time, Accuracy)
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% use Knn after princomp
demension=10:10:200;
Accuracyprincomp=[];
for iii=1:20
    %totaldata=[traindata;testdata];
    [COEFF,SCORE,LATENT] = princomp(traindata);
    percent = cumsum(LATENT)./sum(LATENT);
    NumberOfDemension=find(percent>0.99);
    chooseNumber=NumberOfDemension(1);
    traindatagood=COEFF(:,1:demension(iii));
    traindataReduceGood=traindatachoose*traindatagood;
    validationReduce=validationdata*traindatagood;
    testdataReduceGood=testdata*traindatagood;

    KnnGood=fitcknn(traindataReduceGood,trainclasseschoose,'NumNeighbors',5);
    predictlabel1=predict(KnnGood,validationReduce);
    confmat = confusionmat(validationclasses, predictlabel1);
    Accuracyprincomp(iii) = sum(diag(confmat)) / sum(confmat(:));
end

figure
plot(demension,Accuracyprincomp)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%after choose these two value do last prediction
[COEFF,SCORE] = princomp(traindata);
traindatagood=COEFF(:,1:chooseNumber);
traindataReduceGood=traindata*traindatagood;
testdataReduceGood=testdata*traindatagood;

KnnGood=fitcknn(traindataReduceGood,trainclasses,'NumNeighbors',5);
predictlabel1=predict(KnnGood,testdataReduceGood);
confmat = confusionmat(testclasses, predictlabel1);
resultOfprincomp = sum(diag(confmat)) / sum(confmat(:))
matrixOfprincomp = confmat

%%%%%%%%%%%%%%%%%%%%%%%%%%without princomp
Knnorig=fitcknn(traindata,trainclasses,'NumNeighbors',5);
predictlabel1=predict(Knnorig,testdata);

confmat = confusionmat(testclasses, predictlabel1);
resultWithoutprincomp = sum(diag(confmat)) / sum(confmat(:))
matrixWithoutprincomp = confmat
