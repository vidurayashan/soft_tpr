import torch 
import logging 

logging.basicConfig(level=logging.INFO)

class EarlyStopper: 
    def __init__(self, patience: int, min_delta: float) -> None: 
        self.patience = patience 
        self.min_delta = min_delta
        self.counter = 0 
        self.min_val_loss = float('inf')
    
    def __call__(self, validation_loss: torch.Tensor) -> bool: 
        validation_loss = validation_loss.item() 
        if validation_loss < self.min_val_loss: 
            self.min_val_loss = validation_loss
            self.counter = 0 
        elif validation_loss > (1-self.min_delta)*(self.min_val_loss): 
            self.counter += 1 
            logging.info(f'********Counter incremented {self.counter}' +  
                         f'curr val loss: {validation_loss}, min: {self.min_val_loss}*******')
            if self.counter >= self.patience: 
                logging.info(f'******** PATIENCE EXCEEDED. Stopping training {self.counter}' +  
                         f'curr val loss: {validation_loss}, min: {self.min_val_loss}*******')
                return True 
        return False 
    