%This function gets the SIFT features from the input image dataset
function [Data] = SIFTFeatureExtraction(imageDataset)
    %set the number of SIFT features to use to 50
    SIFTNUM = 50;

    %% Get dataset size
    datasetSize = size(imageDataset.Files);
    %Data = zeros(datasetSize(1), SIFTNUM*128);
    
    %% Organize Data
    for i = 1:datasetSize(1)
        
        %Convert to grayscale
        image = rgb2gray(imread(imageDataset.Files{i}));
        image = histeq(image);
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
        if(size(imFeatures, 1) > SIFTNUM)
            imFeatures((SIFTNUM + 1):end, :) = [];
        end
        if(size(imFeatures, 1) < SIFTNUM)
            pad = zeros(SIFTNUM - size(imFeatures, 1), 128);
            imFeatures = [imFeatures; pad];
        end
    
        %Put in custom table
        Data(i, :) = reshape(imFeatures, [1, numel(imFeatures)]);
    
        %Must clear this varible to reuse as the dimensions changed
        imFeatures = 0;

    end
end