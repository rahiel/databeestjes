function [ relevancies ] = predictionToRelevancies(predictions,actuals)

% Determine the inverse of sorting the predictions
% As the actuals are sorted, but we are interested
% in the original ordering
[~,I] = sortrows(predictions);
unsorted = 1:length(predictions);
indices(I) = unsorted;

% Concatenate actual relevancies to the predictions
relevancies = [predictions,actuals(indices,3)];

end
