%The following function trains and test a network for SIFT feature input
function [AccuracyTrain, AccuracyTest] = SIFTNetwork(Data)
    %% Split the data to training and testing
    %get number of observations to go in each data part
    numFeatures = size(Data, 1);
    numFeaturesTrain = floor(0.9*numFeatures);
    %Create the indices for the amounts to go to train and test
    idx = randperm(numFeatures);
    idxTrain = idx(1:numFeaturesTrain);
    idxTest = idx(numFeaturesTrain+1:end);
    %Partion the table for training and testing from the random indices
    TableTrain = {Data(idxTrain, :), imageDataset.Labels(idxTrain, :)};
    TableTest = {Data(idxTest, :), imageDataset.Labels(idxTest, :)};
    
    %% Create layers and options for training
    
    % Putting batch normalization after each fully connected layer
    % Setting to start with 32 nodes, then 8, then 2 for the final binary
    % classification
    % Final activation layer before the classification
    layers = [
        featureInputLayer(size(Data, 2), 'Normalization','zscore') ...
        batchNormalizationLayer ...
        fullyConnectedLayer(32) ...
        batchNormalizationLayer ...
        fullyConnectedLayer(8) ...
        batchNormalizationLayer ...
        fullyConnectedLayer(2) ...
        batchNormalizationLayer ...
        reluLayer ...
        classificationLayer];
    
    % Setting max epochs to 100
    % Mini-batch size of 32 selected
    % Show plots while learning
    options = trainingOptions('sgdm', ...
        MaxEpochs = 100,...
        InitialLearnRate = 2e-3, ...
        MiniBatchSize = 32, ...
        Verbose = false, ...
        Plots = 'training-progress');
    
    %Train the network
    [net, netInfo] = trainNetwork(TableTrain{:},layers,options);
    
    %% Check accuracy
    % Get the training accuracy
    AccuracyTrain = netInfo.TrainingAccuracy(end);
    
    % Get the testing accuracy
    NetworkPredict = classify(net, TableTest{1});
    LabelTest = TableTest{2};
    AccuracyTest = sum(NetworkPredict == LabelTest)/numel(LabelTest);
end