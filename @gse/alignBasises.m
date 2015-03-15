function alignBasises(this)
  reducedDimension = this.reducedDimension;
  weightedLinearSpacesProjections = cell(this.sampleSize);
  diagonalLinearSpaceProjections = cell(this.sampleSize,1);
  eyeReducedDimension = eye(reducedDimension);

  for pointIndex1 = 1:this.sampleSize
    for pointIndex2 = pointIndex1:this.sampleSize
      if pointIndex1 == pointIndex2
        weightedLinearSpacesProjections{pointIndex1, pointIndex2} = eyeReducedDimension;
      else
        weightedLinearSpacesProjections{pointIndex1, pointIndex2} = this.kernels(pointIndex1,pointIndex2) * ...
            (this.linearSpacesProjections{pointIndex1, pointIndex2});
        weightedLinearSpacesProjections{pointIndex2, pointIndex1} = weightedLinearSpacesProjections{pointIndex1, pointIndex2}';
      end

    end
    diagonalLinearSpaceProjections{pointIndex1} = eyeReducedDimension*sum(this.kernels(pointIndex1, :));
  end

  phi1 = cell2mat(weightedLinearSpacesProjections);
  phi0 = blkdiag(diagonalLinearSpaceProjections{:});

  % solving generalized eigenvalue problem
  options.disp = 0;
  options.isreal = 1;
  options.issym = 1;
  phi = phi0-phi1;
  phi0 = phi0/sum(sum(phi0));
  [eigenMatrix, ~] = eigs(phi, phi0, reducedDimension, 'SA', options);% W = {v_i}|i=1,n
  for pointIndex = 1:this.sampleSize
    v = eigenMatrix((pointIndex-1)*reducedDimension+1:pointIndex*reducedDimension,:); % v_i
    [U,~,V] = svd(v);
    this.vs{pointIndex} = U*V';
    this.projectionJacobians{pointIndex} = this.localPCs{pointIndex} * this.vs{pointIndex}; % H(X_i)
    vTv{pointIndex} = this.vs{pointIndex}' * this.vs{pointIndex};
  end
  this.vTv = vTv;
end