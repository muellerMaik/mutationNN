class Connection:
    def __init__(self, INPUT_ID: int, OUTPUT_ID: int, WEIGHT: float):
        self.INPUT_ID = INPUT_ID
        self.OUTPUT_ID = OUTPUT_ID
        self.WEIGHT = WEIGHT
        
    def mutate_Weight(self, mutation_Value: float):
        self.WEIGHT *= mutation_Value