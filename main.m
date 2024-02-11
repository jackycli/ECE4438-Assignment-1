%% Split the data to training and testing
%get number of observations to go in each data part
numFeatures = size(Data,1);
numFeaturesTrain = floor(0.8*numFeatures);
numFeaturesTest = numFeatures - numFeaturesTrain;
%Create the indices for the amounts to go to train and test
idx = randperm(numFeatures);
idxTrain = idx(1:numFeaturesTrain);
idxTest = idx(numFeaturesTrain+1:end);
%Partion the table for training and testing from the random indices
TableTrain = {Data(idxTrain, :), imageDataset.Labels(idxTrain, :)};
TableTest = {Data(idxTest, :), imageDataset.Labels(idxTest, :)};


%% Create layers and options for training
% Used the deep Network Desiginer to generate code for the layers
% It is imported from the .mat file

options = trainingOptions('sgdm', ...
    'MaxEpochs',200,...
    'InitialLearnRate',2e-3, ...
    'Verbose',false, ...
    'Plots','training-progress');

layers = [featureInputLayer(size(Data,2))
     batchNormalizationLayer
     fullyConnectedLayer(32)
     batchNormalizationLayer
     fullyConnectedLayer(8)
     batchNormalizationLayer
     fullyConnectedLayer(2)
     batchNormalizationLayer
     reluLayer
     classificationLayer
     ];

net = trainNetwork(TableTrain{:},layers,options);

%% Check accuracy
% Get the training accuracy

NetworkPredict = classify(net, TableTest{1});
LabelTest = TableTest{2};
AccuracyTest = sum(NetworkPredict == LabelTest)/numel(LabelTest);