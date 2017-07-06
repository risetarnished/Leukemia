clear;
clc;
close all;

load('golubsmall.mat');
traindata = traindata';
% trainclasses = trainclasses';
% testdata = testdata';
% testclasses = testclasses';

[COEFF, SCORE, LATENT] = princomp(traindata);
p1 = cumsum(LATENT) ./ sum(LATENT);
[coeff, score, latent] = pca(traindata);
p2 = cumsum(latent) ./ sum(latent);
disp([p1(1:37) p2]);
