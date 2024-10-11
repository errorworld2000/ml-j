import math
import random

class NeuronNet:
    def __init__(self,num_inputs,num_hidden,num_outputs,hidden_layer_weights=None,hidden_layer_bias=None,output_layer_weights=None,output_layer_bias=None,learning_rate=0.5):
        super().__init__()
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.num_hidden=num_hidden
        self.hidden_layer = NeuronLayer(num_hidden,hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs,output_layer_bias)
        self.init_weight(num_inputs,self.hidden_layer,hidden_layer_weights)
        self.init_weight(num_hidden,self.output_layer,output_layer_weights)
        self.learning_rate=learning_rate
    
    def init_weight(self,num_inputs,layer,weight):
        if weight is None:
            weight=[[random.uniform(-1, 1) for i in range(num_inputs)]for j in range(len(layer.neurons))]
        self.weights=weight
        for i in range(len(layer.neurons)):
            layer.neurons[i].weights=self.weights[i]
    
    def inspect(self):
        print("------Neuron Net------ ")
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('* Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------end of Neuron Net------')
        
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)
        
    def train(self,training_inputs,training_outputs):
        self.feed_forward(training_inputs)
        pd_error_wrt_output_neurons_total_net_input=[]
        for o in range(len(self.output_layer.neurons)):
            pd_error_wrt_output_neurons_total_net_input.append(self.output_layer.neurons[o].pd_error_wrt_total_net_input(training_outputs[o]))
        pd_error_wrt_hidden_neurons_total_net_input=[]
        for h in range(len(self.hidden_layer.neurons)):
            pd_error_wrt_hidden_neurons_total_net_output=0
            for o in range(len(self.output_layer.neurons)):
                pd_error_wrt_hidden_neurons_total_net_output+=pd_error_wrt_output_neurons_total_net_input[o]*self.output_layer.neurons[o].weights[h]
            pd_error_wrt_hidden_neurons_total_net_input.append(pd_error_wrt_hidden_neurons_total_net_output*self.hidden_layer.neurons[h].pd_total_net_output_wrt_total_net_input())
            
        for o in range(len(self.output_layer.neurons)):
            self.output_layer.neurons[o].bias-=self.learning_rate*pd_error_wrt_output_neurons_total_net_input[o]
            for w in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight=pd_error_wrt_output_neurons_total_net_input[o]*self.output_layer.neurons[o].pd_total_net_input_wrt_weight(w)
                self.output_layer.neurons[o].weights[w]-=self.learning_rate*pd_error_wrt_weight
        for h in range(len(self.hidden_layer.neurons)):
            self.hidden_layer.neurons[h].bias-=self.learning_rate*pd_error_wrt_hidden_neurons_total_net_input[h]
            for w in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight=pd_error_wrt_hidden_neurons_total_net_input[h]*self.hidden_layer.neurons[h].pd_total_net_input_wrt_weight(w)
                self.hidden_layer.neurons[h].weights[w]-=self.learning_rate*pd_error_wrt_weight
        
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error
        
class NeuronLayer:
    def __init__(self,num_neurons,bias):
        super().__init__()
        self.neurons=[]
        for i in range(num_neurons):
            self.neurons.append(Neuron(bias[i]))
            
    def inspect(self):
        print("Neuron Count: ",len(self.neurons))
        for n in range(len(self.neurons)):
            print("Neuron: ",n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.neurons[n].bias)
    
    def feed_forward(self,inputs):
        self.outputs=[]
        for neuron in self.neurons:
            self.outputs.append(neuron.calculate_output(inputs))
        return self.outputs

class Neuron:
    def __init__(self,bias):
        super().__init__()
        self.bias = bias if bias else random.uniform(-1, 1)
        self.weights=[]
        
    def calculate_output(self,inputs):
        self.inputs=inputs
        self.output=self.squash(self.calculate_total_net_input())
        return self.output
    
    def calculate_total_net_input(self):
        total=0
        for i in range(len(self.inputs)):
            total+=self.inputs[i]*self.weights[i]
        return total+self.bias

    def squash(self,total_net_input):
        return 1/(1+math.exp(-total_net_input))
    
    def calculate_error(self,target):
        return 0.5*(self.output-target)**2/2
    
    def pd_error_wrt_total_net_input(self,target):
        return self.pd_error_wrt_total_net_output(target)*self.pd_total_net_output_wrt_total_net_input()
    
    def pd_error_wrt_total_net_output(self,target):
        return self.output-target
    
    def pd_total_net_output_wrt_total_net_input(self):
        return self.output*(1-self.output)
    def pd_total_net_input_wrt_weight(self,index):
        return self.inputs[index]
