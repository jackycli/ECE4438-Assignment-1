%The following function displays accuracy data about both networks to the
%console
function [] = DisplayInfo(AccuracyTrain, AccuracyTest, AccuracyTrainRaw, AccuracyTestRaw)
    %% Print the information to the console
    formatSpec = "" + ...
        "###### Network With SIFT Feature Input ######\n" + ...
        "####     Training accuracy: %3.2f%%.     ####\n" + ...
        "####     Testing accuracy : %3.2f%%.      ####\n" + ...
        "#############################################" + ...
        "\n" + ...
        "\n" + ...
        "######  Network With Raw Image Input   ######\n" + ...
        "####     Training accuracy: %3.2f%%.     ####\n" + ...
        "####     Testing accuracy : %3.2f%%.      ####\n" + ...
        "#############################################";
    sprintf(formatSpec, AccuracyTrain, AccuracyTest*100, AccuracyTrainRaw, AccuracyTestRaw*100)

end