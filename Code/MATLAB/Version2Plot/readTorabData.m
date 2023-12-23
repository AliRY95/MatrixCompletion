function observedData = readTorabData( pathToData )
    trainDataRead = readmatrix( pathToData + "/ratings_matrix_MovieLens.csv" );
    trainDataRead = trainDataRead( 2:end, 2:end );
    observedData = sparse( trainDataRead' );
end