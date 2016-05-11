function [ nDCG ] = nDCG( predictions, actuals, queryIds )
% nDCG Calculates the normalized discounted cummulative gain
% predictions are the predictions by the algorithm (on your test set)
% actuals should be the list of properties with their scores (from your test set)
% queryIds is a list of unique queryIds
%
% Example usage (not real matlab code):
%
%   data = load("dataset.csv")
%   [trainingset, testset] = createSets(data)
%   algorithm = train(trainingset)
%
%   predictions = algorithm.apply(testset)
%   answers = testset
%   queryIds = unique(data(0,:)) // first column are the query id's
%
%   score = nDCG(predictions, answers, queryIds);
  
relevancies = predictionToRelevancies(predictions,actuals);
optimal = sortrows(actuals,[1,-3]);

    
[~,pos] = ismember(queryIds,predictions(:,1),'legacy');
start = 1;
for i = 1:length(pos)
    finish = pos(i);
    number = (finish - start) + 1;
    
    ids(start:finish) = 1:number;
    
    start = pos(i)+1;
end

ids = ids';

relevancies = [relevancies,ids];
optimal = [optimal,ids];
    
dcg = DCG(relevancies, ids);
optDcg = DCG(optimal, ids);

dcgSums = accumarray(dcg(:,1),dcg(:,2),[],@sum);
dcgOptSums = accumarray(optDcg(:,1),optDcg(:,2),[],@sum);

dcgRatios = dcgSums ./ dcgOptSums;

dcgRatios(isnan(dcgRatios)) = [];

nDCG = sum(dcgRatios) / length(dcgRatios);

end

