function [ trainData, testData ] = readMovieLensData( pathToData, num )
    if ( num == 0 )
        trainDataRead = tdfread( pathToData + "/u.data", '\t' );
        trainDataRead = struct2cell( trainDataRead );
        trainData = sparse( trainDataRead{2}, trainDataRead{1}, trainDataRead{3} );
        testData = sparse( size( trainData, 1 ), size( trainData, 2 ) );
    else
        trainDataRead = tdfread( pathToData + "/u" + int2str(num) + ".base", '\t' );
        trainDataRead = struct2cell( trainDataRead );
        trainData = sparse( trainDataRead{2}, trainDataRead{1}, trainDataRead{3} );
        testDataRead = tdfread( pathToData + "/u" + int2str(num) + ".test", '\t' );
        testDataRead = struct2cell( testDataRead );
        testData = sparse( size( trainData, 1 ), size( trainData, 2 ) );
        indices = sub2ind( size( testData ), testDataRead{2}, testDataRead{1} );
        testData( indices ) = testDataRead{3};
    end
    
end