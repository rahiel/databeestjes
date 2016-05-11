function [ dcg ] = DCG(relevancies, ids)
% Calculates the Discounted Cummulative Gain

% see http://en.wikipedia.org/wiki/Discounted_cumulative_gain
% below is an array implementation of the for loop below
dcg = relevancies(:,1);
dcg(:,2) = (2.^relevancies(:,3) - 1) ./ log2(1+ids);

% for i = 1:k
%     % see http://en.wikipedia.org/wiki/Discounted_cumulative_gain
%     dcg = dcg + (((2.^relevancies(i)) - 1.0) / log2(1 + i));
% end

end

