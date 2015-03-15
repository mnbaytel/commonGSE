function kernels = calculateKernels(model, points, otherPoints)
  if nargin < 3
    otherPoints = points;
  end
  type = model.KernelType;
  width = model.KernelWidth;
  maxDist = model.EuclideanMetricsThreshold;
  switch lower(type)
    case {'gaussian', 'gaus'}
      euclideanDist = dist(points, otherPoints');
      kernels = (euclideanDist < maxDist) .* exp(-width * euclideanDist);
    case {'hat'}
      euclideanDist = dist(points, otherPoints');
      kernels = exp(1 ./ ((width * euclideanDist).^2 - 1)) .* (width * euclideanDist < 1);
    otherwise
      exception = MException('gse:calculateKernelsBadOption', 'Unknown kernel name');
      throw(exception);
  end
end