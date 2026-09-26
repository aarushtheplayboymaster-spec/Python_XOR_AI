import math
class XOR_AI:
    def __init__(self):
        #randam weights and biases
        self.w1 = 0.1
        self.w2 = 0.2
        self.w3 = 0.4
        self.w4 = 0.3
        self.w5 = 0.7
        self.w6 = 0.5
        self.bias1 = 0.1
        self.bias2 = 0.1
        self.bias3 = 0.1
#sigmoid formula :- 1 / 1 + e ^ -z (this formula is used to convert any real-valued number into predictable value b/w 0 - 1 )
    def sigmoid(self,z):
        return 1.0 / (1.0 + math.exp(-z))
#sigmoid Derivartive formula :- z x (1 - z) (this formula is used for Gradient descent to find the AI's mistake and improve weights and biases )
    def sigmoid_Derivarive(self,z):
        return z * (1.0 - z) 
#neural network this usage a formula of (n = w x input + b) then it is passed into sigmoid()
    def neural_network(self,input1,input2):
        self.input1 = input1
        self.input2 = input2
        znode1 = (self.w1 * self.input1) + (self.w3 * self.input2) + self.bias1
        self.node1 = self.sigmoid(znode1)
        znode2 = (self.w2 * self.input1) + (self.w4 * self.input2) + self.bias2
        self.node2 = self.sigmoid(znode2)
        zoutput = (self.node1 * self.w5) + (self.node2 * self.w6) + self.bias3
        self.output = self.sigmoid(zoutput)
        return self.output
#backpropogation
    def backpropogation(self,targate):
        learning_rate = 0.5
        output = self.output

        error = output - targate #it gets the scroe of how much trach was the prediction

        DSoutput = error * self.sigmoid_Derivarive(output)
        DSh1 = DSoutput * self.w5 * self.sigmoid_Derivarive(self.node1)
        DSh2 = DSoutput * self.w6 * self.sigmoid_Derivarive(self.node2)

        # Updateing weights(it goes from the last node's weights and biases to first nodes weights and biases)

        #update's weight5 and weight6 and biases3
        self.w5 -= learning_rate * DSoutput * self.node1
        self.w6 -= learning_rate * DSoutput * self.node2
        self.bias3 -= learning_rate * DSoutput 

        #update's weight1 and weight3 and biases1
        self.w1 -= learning_rate * DSh1 * self.input1
        self.w3 -= learning_rate * DSh1 * self.input2
        self.bias1 -= learning_rate * DSh1

        #update's weight2 and weight4 and biases2
        self.w2 -= learning_rate * DSh2 * self.input1
        self.w4 -= learning_rate * DSh2 * self.input2
        self.bias2 -= learning_rate * DSh2

dataset = [
    ([0,0],0),
    ([0,1],1),
    ([1,0],1),
    ([1,1],0),
    ([1,2],1),
    ([2,1],1),
    ([2,2],0),
    ([1,3],1),
    ([3,1],1),
    ([3,3],0),
    ]

neural = XOR_AI()
loops = 10000
for loop in range(loops):
    for (input1,input2), targat in dataset:
        neural.neural_network(input1=input1,input2=input2)
        neural.backpropogation(targat)
for (input1, input2), target in dataset:
    pred = neural.neural_network(input1, input2)
    print(f"Input: [{input1}, {input2}] -> Predicted: {pred:.4f} (Target: {target})")