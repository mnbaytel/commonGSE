function [principalComponentsPerPoint, eigenValuesPerPoint] = ...
  calculateWeightedPCA(this, points, weights, otherPoints, wantedDimension)
  if (nargin < 4) || any(size(otherPoints) == 0)
    otherPoints = points;
  end

  principalComponentsPerPoint = cell(size(weights, 1), 1);
  eigenValuesPerPoint = cell(size(weights, 1), 1);
  if nargin < 5
    wantedDimension = this.reducedDimension;
  end

  for pointIndex = 1:length(principalComponentsPerPoint)
    localWeights = weights(pointIndex, :)';
    localIndeces = localWeights > 0;
    localWeights = localWeights(localIndeces);
    localWeightedPoints = otherPoints(localIndeces,:).*repmat(sqrt(localWeights), 1, size(points, 2));
    localWeightedPoints = localWeightedPoints - repmat(mean(localWeightedPoints), size(localWeightedPoints,1),1);
    [principalComponentsPerPoint{pointIndex}, eigenValuesPerPoint{pointIndex}] = ...
      eigs(localWeightedPoints' * localWeightedPoints, wantedDimension, 'LA');
  end

end