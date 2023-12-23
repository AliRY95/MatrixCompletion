function predictedData = scaleData( observedData, predictedData )
    indices = find( observedData );
    observedData = observedData( indices );
    % discreteVals = unique( observedData );
    % numDiscreteVals = size( discreteVals );
    minObservedData = min( min( observedData ) );
    maxObservedData = max( max( observedData ) );
    meanObservedData = mean( observedData );
    stdObservedData = std( observedData );
    
    minPredictedData = min( predictedData( indices ) );
    rangePredictedData = max( predictedData( indices ) ) - minPredictedData;
    
    % predictedData = ( predictedData - minPredictedData ) / rangePredictedData;
    % predictedData = predictedData * ( maxObservedData - minObservedData ) ...
    %                 + minObservedData;
    predictedData = ( predictedData - meanObservedData ) / stdObservedData;
    % predictedData = round( predictedData );
end