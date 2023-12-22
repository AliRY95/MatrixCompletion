function [NewMAT, ier] = MatrixCompletion(A, N, lambda_tol, tol, display)

Mask = full( A ~= 0 );
ier = 0;
min_lambda = 0;
max_lambda = sum(svd(A))*1.1;
NewMAT = A;
lambda = inf;
lambda_prev = 0;
err = inf;
Counter = 0;
Converge=0;
while ( ( err > tol ) || ( abs( lambda - lambda_prev ) > lambda_tol ) )
    Counter = Counter + 1;
    lambda_prev = lambda;
    lambda = ( min_lambda + max_lambda ) / 2;
    [NewMAT, Error] = MatApproxNuclear( A.*Mask, NewMAT, Mask, lambda, N, tol, display);
    err = Error(end);
    if Error(end) > tol
        min_lambda=lambda;
    else
        max_lambda=lambda;
    end
    if abs(lambda-lambda_prev) < lambda_tol
        Converge=Converge+1;
    end
    if Converge>4
        if display
            fprintf('Looks like the algorithm failed to converge. There are two options: \n');
            fprintf('Initial matrix norm is too small or more iterations are needed (larger N) \n');
        end
        ier=1;
        return;
    end    
end

end

        