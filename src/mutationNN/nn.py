from .enum_Class import Activation_Function
from .neuron import Neuron
from .connection import Connection
from random import uniform, randint, choice
from copy import deepcopy

class NN:
    def __init__(self, INPUT_SIZE: int, OUTPUT_SIZE: int, MUTATION_CHANCE: float = 0.25, FULLY_CONNECTED: bool = False) -> None:
        self.INPUT_SIZE = INPUT_SIZE
        self.OUTPUT_SIZE  = OUTPUT_SIZE
        self.MUTATION_CHANCE = MUTATION_CHANCE
        
        self.MAX_LAYER = 1
        self.NEURONS = []
        self.CONNECTIONS = []

        for i in range(self.INPUT_SIZE + self.OUTPUT_SIZE):
            neuron_id = len(self.NEURONS)
            bias = uniform(-10, 10)
            if i < self.INPUT_SIZE:
                self.add_Neuron(
                    0,
                    Activation_Function.LINEAR,
                    "INPUT"
                )
            else:
                self.add_Neuron(
                    bias,
                    Activation_Function.LINEAR,
                    "OUTPUT"
                )

        if FULLY_CONNECTED:
            for input_id in range(self.INPUT_SIZE):
                for output_id in range(self.INPUT_SIZE, self.INPUT_SIZE+self.OUTPUT_SIZE):
                    self.add_Connection(
                        input_id,
                        output_id,
                        uniform(-1, 1)
                    )
        else:
            for input_id in range(self.INPUT_SIZE):
                for output_id in range(self.INPUT_SIZE, self.INPUT_SIZE+self.OUTPUT_SIZE):
                    if self.MUTATION_CHANCE > uniform(0, 1):
                        self.add_Connection(
                            input_id,
                            output_id,
                            uniform(-1, 1)
                        )

    def __add_Neuron(self, bias: float, activation: Activation_Function = None, category: str = "HIDDEN") -> None:
        match category:
            case "HIDDEN":
                newLayer = randint(1, self.MAX_LAYER)
                newID = len(self.NEURONS)
                for ne in self.NEURONS:
                    if ne.LAYER == newLayer and ne.CATEGORY == "OUTPUT":
                        ne.LAYER += 1
                
                self.NEURONS.append(
                    Neuron(
                        ID=newID,
                        LAYER=newLayer,                    
                        CATEGORY=category,
                        ACTIVATION=Activation_Function(randint(0, len(Activation_Function)-1)).name,
                        BIAS=bias
                    )
                )
                
                self.MAX_LAYER += 1 if newLayer == self.MAX_LAYER else 0
                
                if len(self.CONNECTIONS) == 0:
                    return
                   
                validConnection = []
                for i, conn in enumerate(self.CONNECTIONS):
                    if self.NEURONS[conn.INPUT_ID].LAYER < newLayer and self.NEURONS[conn.OUTPUT_ID].LAYER > newLayer:
                        validConnection.append(i)
                if len(validConnection) > 0:
                    pickedConnection = validConnection[randint(0, len(validConnection)-1)]
                
                    self.add_Connection(self.CONNECTIONS[pickedConnection].INPUT_ID, newID)
                    self.add_Connection(newID, self.CONNECTIONS[pickedConnection].OUTPUT_ID, self.CONNECTIONS[pickedConnection].WEIGHT)
                    self.pop_Connection(pickedConnection)
                
            case "INPUT":
                self.NEURONS.append(
                    Neuron(
                        ID=len(self.NEURONS),
                        LAYER=0,                    #Alle Inputs sind IMMER layer 0
                        CATEGORY=category,
                        ACTIVATION=activation,
                        BIAS=bias
                    )
                )
            case "OUTPUT":
                self.NEURONS.append(
                    Neuron(
                        ID=len(self.NEURONS),
                        LAYER=self.MAX_LAYER,       #Alle Outputs sind IMMER layer MAX_LAYER
                        CATEGORY=category,
                        ACTIVATION=activation,
                        BIAS=bias
                    )
                )
                
    def __add_Connection(self, INPUT_ID: int, OUTPUT_ID: int, WEIGHT: float = 1.0) -> None:
        self.CONNECTIONS.append(
            Connection(
                INPUT_ID,
                OUTPUT_ID,
                WEIGHT
            )
        )
    
    def __pop_Connection(self, connection_position: int) -> None:
        self.CONNECTIONS.pop(connection_position)
        
    def __pop_Neuron(self, neuron_id: int) -> None:
        self.NEURONS.pop(neuron_id)
        self.CONNECTIONS = [conn for conn in self.CONNECTIONS if conn.INPUT_ID != neuron_id and conn.OUTPUT_ID != neuron_id]
        for updated_id, ne in enumerate(self.NEURONS):
            if ne.ID != updated_id:
                for conn in self.CONNECTIONS:
                    if conn.INPUT_ID == ne.ID:
                        conn.INPUT_ID = updated_id
                    elif conn.OUTPUT_ID == ne.ID:
                        conn.OUTPUT_ID = updated_id
                ne.ID = updated_id
            
    def __clear_Network(self) -> None:
        for ne in self.NEURONS:
            ne.update_Value(0.0)
            
    def __feed_Forward(self) -> list:
        outputs = []
        currentLayer = 0
        skipSetConn = set()
        skipSetNe = set()
        while currentLayer <= self.MAX_LAYER:
            for i, ne in enumerate(self.NEURONS):
                if currentLayer == ne.LAYER and i not in skipSetNe:
                    skipSetNe.add(i)
                    ne.add_Bias()
                    ne.apply_Activation()
                    
            for i, conn in enumerate(self.CONNECTIONS):
                if i in skipSetConn:
                    continue
                    
                elif self.NEURONS[conn.INPUT_ID].LAYER == currentLayer:
                    skipSetConn.add(i)
                    self.NEURONS[conn.OUTPUT_ID].add_Value(self.NEURONS[conn.INPUT_ID].VALUE, conn.WEIGHT)
            currentLayer += 1
        
        for ne in self.NEURONS:
            if ne.CATEGORY == "OUTPUT":
                outputs.append(ne.VALUE)
        
                
        return outputs

    def process_Input(self, inputList: list) -> list:
        counter = 0
        self.clear_Network()
        for ne in self.NEURONS:
            if ne.CATEGORY == "INPUT":
                ne.update_Value(inputList[counter])
                counter += 1
            elif counter == len(inputList):
                break
                
        return self.feed_Forward()
        
    def process_Static_Input(self, staticInputList: list) -> list:
        outputList = []
        for inputList in staticInputList:        
            outputList.append(self.process_Input(inputList))
        return outputList
               
    def mutate(self, must_mutate: bool = False) -> None:
        if not must_mutate:
            for ne in self.NEURONS:
                if self.MUTATION_CHANCE > uniform(0, 1):
                    ne.mutate_Bias(uniform(-2, 2)) if uniform(0,1) > 0.5 else ne.mutate_Activation(randint(1, len(Activation_Function) - 1))
                    
            for conn in self.CONNECTIONS:
                if self.MUTATION_CHANCE > uniform(0, 1):
                    conn.mutate_Weight(uniform(-2, 2))
                    
            has_Hidden_Neuron = len([i for i in range(len(self.NEURONS)) if self.NEURONS[i].CATEGORY != "INPUT" and self.NEURONS[i].CATEGORY != "Output"]) > 0
            if self.MUTATION_CHANCE > uniform(0, 1):
                must_mutate = False
                self.add_Neuron(uniform(-10, 10))
                
            elif self.MUTATION_CHANCE > uniform(0, 1) and has_Hidden_Neuron:
                self.pop_Neuron(randint(0, len(self.NEURONS)-1))
                
            if self.MUTATION_CHANCE > uniform(0, 1):
                possibleConnections = set()
                currentConnections = set([(conn.INPUT_ID, conn.OUTPUT_ID) for conn in self.CONNECTIONS])
                for input_id, ne_in in enumerate(self.NEURONS):
                    for output_id, ne_out in enumerate(self.NEURONS):
                        if ne_in.LAYER < ne_out.LAYER and (input_id, output_id) not in currentConnections:
                            possibleConnections.add((input_id, output_id))
                if len(possibleConnections) > 0:
                    newConnection = choice(list(possibleConnections))
                    self.add_Connection(newConnection[0], newConnection[1], uniform(-1, 1))
            elif self.MUTATION_CHANCE > uniform(0, 1) and len(self.CONNECTIONS) > 1:
                self.pop_Connection(randint(0, len(self.CONNECTIONS)-1))
            return
            
        while must_mutate:
            for ne in self.NEURONS:
                if self.MUTATION_CHANCE > uniform(0, 1):
                    must_mutate = False
                    ne.mutate_Bias(uniform(-2, 2)) if uniform(0,1) > 0.5 else ne.mutate_Activation(randint(1, len(Activation_Function) - 1))
                    
            for conn in self.CONNECTIONS:
                if self.MUTATION_CHANCE > uniform(0, 1):
                    must_mutate = False
                    conn.mutate_Weight(uniform(-2, 2))
                    
            has_Hidden_Neuron = len(self.NEURONS) > (self.INPUT_SIZE+self.OUTPUT_SIZE)
            if self.MUTATION_CHANCE > uniform(0, 1):
                must_mutate = False
                self.add_Neuron(uniform(-10, 10))
            elif self.MUTATION_CHANCE > uniform(0, 1) and has_Hidden_Neuron:
                must_mutate = False
                self.pop_Neuron(randint(0, len(self.NEURONS)-1))
                
            if self.MUTATION_CHANCE > uniform(0, 1):
                must_mutate = False
                possibleConnections = set()
                currentConnections = set([(conn.INPUT_ID, conn.OUTPUT_ID) for conn in self.CONNECTIONS])
                for input_id, ne_in in enumerate(self.NEURONS):
                    for output_id, ne_out in enumerate(self.NEURONS):
                        if ne_in.LAYER < ne_out.LAYER and (input_id, output_id) not in currentConnections:
                            possibleConnections.add((input_id, output_id))
                if len(possibleConnections) > 0:
                    newConnection = choice(list(possibleConnections))
                    self.add_Connection(newConnection[0], newConnection[1], uniform(-1, 1))
            elif self.MUTATION_CHANCE > uniform(0, 1) and len(self.CONNECTIONS) > 1:
                must_mutate = False
                self.pop_Connection(randint(0, len(self.CONNECTIONS)-1))
                
    def copy_NN(self, NN) -> None:
        self.MAX_LAYER = deepcopy(NN.MAX_LAYER)
        self.NEURONS = deepcopy(NN.NEURONS)
        self.CONNECTIONS = deepcopy(NN.CONNECTIONS)
        self.mutate(True)