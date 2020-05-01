LIBS += ../bpnn/bpnn.cpp
LIBS += ../logistics/logistics.cpp
LIBS += ../perceptron/perceptron.cpp
LIBS += ../svm/svm.cpp
all:
	g++ -o test_xor test_xor.cpp $(LIBS)
	g++ -o test_bayes test_bayes.cpp ../bayes/bayes.cpp
	g++ -o test_dqn ../dqn/test.cpp ../dqn/dqn.cpp ../bpnn/bpnn.cpp
	g++ -o test_dpg ../policyGradient/test.cpp ../policyGradient/policyGradient.cpp ../bpnn/bpnn.cpp
