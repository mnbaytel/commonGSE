function [decompressedPoints, failedPoints] = decompress(this, newCompressedPoints)
%% Preprocessing
newCompressedPoints = mapminmax('reverse', newCompressedPoints', this.mappingSettingsReducedDimension);
compressedPoints = mapminmax('reverse', this.compressedTrainPoints, this.mappingSettingsReducedDimension);
%% Decompression
compressedKernels = this.calculateKernels(newCompressedPoints', compressedPoints');

% Weighted PCA
compressedLocalPCs = this.calculateWeightedPCA(this.trainPoints, compressedKernels);  % Q(X_i)

compressedKernels = this.adjustKernels(compressedKernels, compressedLocalPCs, this.localPCs);

decompressedPoints = zeros(size(newCompressedPoints, 2), this.originalDimension);

for pointIndex1 = 1:size(newCompressedPoints, 2)
  if isnan(newCompressedPoints(1, pointIndex1))
    decompressedPoints(pointIndex1, :) = NaN(1, this.originalDimension);
  else
  tmpVector = zeros(this.originalDimension,1);
  for pointIndex2 = 1:size(this.trainPoints, 1)
    tmpVector = tmpVector + compressedKernels(pointIndex1, pointIndex2) *(this.trainPoints(pointIndex2,:)' + this.projectionJacobians{pointIndex2} * (newCompressedPoints(:,pointIndex1) - compressedPoints(:, pointIndex2)));
  end
  decompressedPoints(pointIndex1, :) = tmpVector'/sum(compressedKernels(pointIndex1, :));
  end
end

failedPoints = sum(isnan(decompressedPoints(:, 1)));
%% Postprocessing
decompressedPoints = mapminmax('reverse', decompressedPoints', this.mappingSettingsOriginalDimension)';
end