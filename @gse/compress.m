function newCompressedPoints = compress(this, newPoints)
%% Preprocesing
newPoints = mapminmax('apply', newPoints', this.mappingSettingsOriginalDimension)';
compressedPoints = mapminmax('reverse', this.compressedTrainPoints, this.mappingSettingsReducedDimension);

%% Compression

newSampleSize = size(newPoints,1);
newCompressedPoints = zeros(this.reducedDimension, newSampleSize);
newKernels = this.calculateKernels(this.trainPoints, newPoints);
[newLocalPCs, newLocalEigenVals] = this.calculateWeightedPCA(newPoints, newKernels, this.trainPoints);  % Q(X)
newKernels = this.adjustKernels(newKernels, newLocalPCs, this.localPCs); %K_1(X, X_i)
for pointIndex1 = 1:newSampleSize
  kernelSum = sum(newKernels(pointIndex1, :));
  if sum(newKernels(pointIndex1, :) > 0) < this.reducedDimension
    newCompressedPoints(:, pointIndex1) = NaN(this.reducedDimension, 1);
  else
    compressed_KNR = sum(repmat(newKernels(pointIndex1, :), this.reducedDimension, 1) .* compressedPoints, 2) ./ kernelSum;
    decompressed_KNR = sum(repmat(newKernels(pointIndex1, :), this.originalDimension, 1) .* this.trainPoints', 2) ./ kernelSum;
    jacobian_KNR = zeros(this.originalDimension, this.reducedDimension);
    for pointIndex2 = 1:this.sampleSize
      jacobian_KNR = jacobian_KNR + newKernels(pointIndex1, pointIndex2) * this.projectionJacobians{pointIndex2};
    end
    
    v =  newLocalPCs{pointIndex1}' * jacobian_KNR ./ kernelSum;
    [U,~,V] = svd(v);
    v = U*V';
    newCompressedPoints(:, pointIndex1) = compressed_KNR + pinv(v) * newLocalPCs{pointIndex1}' * ...
      (newPoints(pointIndex1,:)' - decompressed_KNR);
  end
end

%% Postprocessing
newCompressedPoints = mapminmax('apply', newCompressedPoints, this.mappingSettingsReducedDimension)';
end