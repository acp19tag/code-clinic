import time, datetime


##################################
# OTHER CLASSES
##################################

class TimeManager:
    
    def __init__(self) -> None:
        
        self.start_time = time.time()
        
    def end(self):
        
        print(f"Script completed in {str(datetime.timedelta(seconds = int(time.time() - self.start_time)))}.")