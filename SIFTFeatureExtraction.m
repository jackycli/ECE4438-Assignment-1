SIFTNUM = 50;

%% Define the location of the dataset
datasetPath = ('C:\Users\JD Herlehy\OneDrive - The University of Western Ontario\Forth Year\4436 Adv Img Proc\Ass 1\WasteBinary');
imageDataset = imageDatastore(datasetPath, "IncludeSubfolders",true, "LabelSource","foldernames");

%% Get dataset size
datasetSize = size(imageDataset.Files);
%Data = zeros(datasetSize(1), SIFTNUM*128);

%% Organize Data
for i = 1:datasetSize(1)
    
    %Convert to grayscale
    image = rgb2gray(imread(imageDataset.Files{i}));
    %Get strongest __ SIFT features
    %Preliminary cut the size with the strongest features
    imgSIFTs = detectSIFTFeatures(image);
    strongest = imgSIFTs.selectStrongest(SIFTNUM);
    %Extract the features
    [imFeatures, valids] = extractFeatures(image, strongest);

    %Since feature extraction on SIFT points yields more features than
    %points due to the multiple orientations at same point, there is more
    %than 200 features unless original points are less than 200
    %Cut down to 200 to match the input layer
    %or pad with zeros to get to 200
    if(length(imFeatures) > SIFTNUM)
        imFeatures((SIFTNUM + 1):end, :) = [];
    end
    if(length(imFeatures) < SIFTNUM)
        pad = zeros(SIFTNUM - length(imFeatures), 128);
        imFeatures = [imFeatures; pad];
    end

    %Put in custom table
    imFeatures = reshape(imFeatures, [1, numel(imFeatures)]);
    Data(i, :) = num2cell(imFeatures);

end

%Append the label data to the cell array
Data = horzcat(Data, num2cell(imageDataset.Labels));
%Convert the cell array to Table form
Data = cell2table(Data);