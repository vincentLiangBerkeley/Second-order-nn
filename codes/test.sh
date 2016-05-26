tests="TestFactorization\nTestNaivePCG\nTestTriSolve\nTestMultiply\nTestPCG\nTestApproximation\nall"
if [ "${tests/$1}" = "$tests" ]; then
    echo -e "Unrecognized test case, valid test cases are:\n"
    echo -e $tests
elif [ "$1" = "all" ]; then
    #statements
    python test.py
else
    python -m unittest -v test.$1
fi