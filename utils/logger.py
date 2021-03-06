from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import time
import os

class Logger:
    """
    Output logs to files and stdout.
    """

    def __init__(self, to_stdout = True, file_path = None, timestamp = True):
        self.to_stdout = to_stdout
        self.file_path = file_path
        self.f_handler = open(file_path, 'a') if file_path else None
        self.f2_handler = open(self.get_dump_file(file_path), 'a') if file_path else None

        self.timestamp = timestamp
    
    def get_dump_file(self, file_path):
        path, name = os.path.split(file_path)
        return os.path.join(path, 'dump_'+name)

    def __del__(self):
        if self.f_handler:
            self.f_handler.close()
        if self.f2_handler:
            self.f2_handler.close()

    def info(self, msg, dump = False):
        if self.timestamp:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            msg = f'{ts} {msg}'
        if self.to_stdout:
            print(msg)
        if self.f_handler:
            self.f_handler.write(msg + '\n')
            self.f_handler.flush()
        
        if dump and self.f2_handler:
            self.f2_handler.write(msg + '\n')
            self.f2_handler.flush()
        
    



        