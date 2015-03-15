trainSize = 250;
surfaceName = 'cylinder';
method = 'GSE';
kernelWidth = 10;
[trainPoints, trainTangentSpace, parametrizationTest] = ...
  generateSampleOnSurface(trainSize, surfaceName);

mapping = gse('LoggingLevel', 'info', 'KernelWidth', kernelWidth);
kernels = mapping.calculateKernels(trainPoints);
reducedDimension = 2;
[principalComponentsPerPoint, eigenValuesPerPoint] = ...
  mapping.calculateWeightedPCA(trainPoints, kernels, [], reducedDimension);

% mapping.train(trainPoints, reducedDimension);
% 
% compressedPoints = mapping.compress(trainPoints);
% decompressedPoints = mapping.decompress(compressedPoints);