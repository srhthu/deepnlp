"""
Record training information such as training loss.
"""

class Reporter:
    def __init__(self):
        ...
    
    def log_step(self):
        ...
    
    def log_epoch(self):
        ...
    
    def log_evaluate(self):
        ...
    