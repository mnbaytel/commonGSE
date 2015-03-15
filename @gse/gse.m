classdef gse < handle
  
  properties (SetAccess = private)
    originalDimension
    reducedDimension
    sampleSize
    
    trainPoints
    compressedTrainPoints
    
    localPCs
    localEigenVals
    linearSpacesProjections 
    
    EuclideanMetricsThreshold % \epsiolon_0
    EigenValuesThreshold % \epsilon_1
    CauchyBinetMetricsThreshold % \epsilon_2
    KernelWidth % \epsilon
    RegularizationLambda % \lambda_{orth}
    KernelType % hat, gaussian
    
    iLogger % log writer
    LoggingLevel % just for info
    
    mappingSettingsOriginalDimension
    mappingSettingsReducedDimension
  end
  
  properties (SetAccess = private, Hidden = true)
    
    kernels
    projectionJacobians
    vs    
    vTv
     
%     currentDelta % current delta for optimization debug 
%     historyDelta % current delta for optimization debug  
%     historyDeltaCompression % current delta compression for optimization debug 
  end
  
  properties (Constant)
    type = 'GSE';
  end
  
  methods (Access = 'public')
    
    function this = gse(varargin)
      % Constructor for GSE
      parser = inputParser; 
      parser.FunctionName = 'GSE:Constructor';
      parser.CaseSensitive = true;
      parser.addOptional('LoggingLevel','Default');
      parser.addOptional('EuclideanMetricsThreshold', Inf);
      parser.addOptional('EigenValuesThreshold', Inf);
      parser.addOptional('CauchyBinetMetricsThreshold', Inf);
      parser.addOptional('KernelWidth', 0.01);
      parser.addOptional('RegularizationLambda', 1e3);
      parser.addOptional('KernelType', 'hat');
      parser.addOptional('reducedDimension', NaN);
      
      parser.parse(varargin{:})
      
      if strcmpi(parser.Results.LoggingLevel, 'Default')
        this.iLogger = logger(this.type);
      else
        this.iLogger = logger(this.type, parser.Results.LoggingLevel);
      end
      this.iLogger.info('Parameters Setting')
      setOptions = parser.Parameters;
      for idx = 1:length(setOptions)
        this.(setOptions{idx}) = parser.Results.(setOptions{idx});
        this.iLogger.info(strcat('Option "', setOptions{idx}, '"', ' set to "', num2str(parser.Results.(setOptions{idx})),'"'))
      end
      this.iLogger.info('Parameters Setting Complete')
    end
    
    function [reconstructedPoints, failedPoints] = reconstruct(model, points)
      [reconstructedPoints, failedPoints] = model.decompress(model.compress(points));
    end
    
    function setProjections(this, newProjections)
      this.projectionJacobians = newProjections;
    end
    
    alignBasises(this)
    calculateJacobianComponent(this, dimensionIndex, kernels, iteration_number)
%      calculateDelta(this, kernels, dimensionIndex)
%      calculateDeltaCompression(this, kernels)
    updatePCs(this, dimension)
    updateVs(this, dimension)
    
    constructCompressedSpace(model, kernels, iteration_number);
    train(model, points, reducedDimension); % , trueTangentSpaces
     
    newCompressedPoints = compress(model, newPoints);
    
    [decompressedPoints, failedPoints] = decompress(model, newPoints);
    
    plotPCs(model);
    
    plotProjectionJacobians(model);
    
  end
  
  methods (Hidden)
    
    kernels = calculateKernels(model, points, weights, otherPoints);
    
    [principalComponentsPerPoint, eigenValuesPerPoint] = calculateWeightedPCA(model, points, weights, otherPoints, wantedDimension)
    
    [kernels, linearSpacesProjections] = adjustKernels(model, kernels, localPCs1, localPCs2)
    
    plotBases(model, bases);
    
  end
end