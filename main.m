%% Split the data to training and testing
%get number of observations to go in each data part
numFeatures = size(Data, 1);
numFeaturesTrain = floor(0.5*numFeatures);
numFeaturesTest = numFeatures - numFeaturesTrain;
%Create the indices for the amounts to go to train and test
idx = randperm(numFeatures);
idxTrain = idx(1:numFeaturesTrain);
idxTest = idx(numFeaturesTrain+1:end);
%Partion the table for training and testing from the random indices
TableTrain = Data(idxTrain, :);
TableTest = Data(idxTest, :);

%% Create layers and options for training
% Used the deep Network Desiginer to generate code for the layers
% It is imported from the .mat file

options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(TableTrain,layers_2,options);