function train(this, trainPoints, reducedDimension)
% GSE train
this.iLogger.info('Training started.')
this.reducedDimension = reducedDimension;
[this.sampleSize, this.originalDimension] = size(trainPoints);
this.iLogger.debug('Sample size: %d', this.sampleSize);
this.iLogger.debug('Input dimension: %d', this.originalDimension);
this.iLogger.debug('Reduced dimension: %d', this.reducedDimension);

this.projectionJacobians = cell(this.sampleSize, 1);
this.vs = cell(this.sampleSize, 1);
this.vTv = cell(this.sampleSize, 1);

%% Preprocessing

this.iLogger.info('Preprocessing: scaling input space')
[trainPoints, this.mappingSettingsOriginalDimension] = mapminmax(trainPoints'); % maybe mapstd better?
trainPoints = trainPoints';
this.trainPoints = trainPoints;

this.iLogger.info('Preprocessing: calculating kernels')

this.kernels = this.calculateKernels(trainPoints);

if this.iLogger.level < logLevel.Info
  this.iLogger.debug('Minimum number of neighbors: %d', min(sum(this.kernels ~= 0)));
  this.iLogger.debug('Maximum number of neighbors: %d', max(sum(this.kernels ~= 0)));
end

this.iLogger.info('Preprocessing: calculating tangent spaces')
% Weighted PCA
[this.localPCs, this.localEigenVals] = this.calculateWeightedPCA(trainPoints, this.kernels);  % Q(X_i), \Lambda(X_i)

this.iLogger.info('Preprocessing: adjusting kernels')
% Adjusting kernels
[this.kernels, this.linearSpacesProjections] = this.adjustKernels(this.kernels, this.localPCs); %K_1(X_i, X_j) and S(X_i, X_j)

if this.iLogger.level < logLevel.Info
  this.iLogger.debug('Min number of neighbors: %d', min(sum(this.kernels ~= 0)));
  this.iLogger.debug('Maximum number of neighbors: %d', max(sum(this.kernels ~= 0)));
end

%% Compression Jacobian calculation
this.iLogger.info('Tangent space alignment')
this.alignBasises

%% Compression calculation
% this.constructCompressedSpace();
% return

this.iLogger.info('Embedding')
% Solving linear system
LHS = cell(this.sampleSize);
RHS = cell(this.sampleSize, 1);

for pointIndex1 = 1:this.sampleSize
  RHStmp = zeros(this.reducedDimension, 1);
  LHSdiag = zeros(this.reducedDimension);
  for pointIndex2 = [1:pointIndex1-1 pointIndex1+1:this.sampleSize]
    tmp = this.kernels(pointIndex1, pointIndex2) * (this.vTv{pointIndex1} + this.vTv{pointIndex2});
    LHS{pointIndex1, pointIndex2} = tmp;
    LHSdiag = LHSdiag - tmp;
    RHStmp = RHStmp + this.kernels(pointIndex1, pointIndex2) * ...
      ((this.projectionJacobians{pointIndex1}' + this.projectionJacobians{pointIndex2}') * ...
      (trainPoints(pointIndex2,:) - trainPoints(pointIndex1,:))');
  end
  LHS{pointIndex1, pointIndex1} = LHSdiag;
  RHS{pointIndex1} = RHStmp;
end

compressedPoints = reshape([cell2mat(LHS); repmat(eye(this.reducedDimension), 1, this.sampleSize)] \ ...
  [cell2mat(RHS); zeros(this.reducedDimension, 1)], this.reducedDimension, this.sampleSize);

%% Postprocessing
[this.compressedTrainPoints, this.mappingSettingsReducedDimension] = mapminmax(compressedPoints);

this.iLogger.info('Training finished.')
end