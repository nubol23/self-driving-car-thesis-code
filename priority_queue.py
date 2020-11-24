from heapq import *
from typing import List, Tuple, Any, Dict, Union, Optional


class PriorityQueue:
    def __init__(self):
        self.h = []
        
    def empty(self) -> bool:
        return False if self.h else True
        
    def push(self, x: Any) -> None:
        heappush(self.h, x)
        
    def pop(self) -> Any:
        return heappop(self.h)
    
    def top(self) -> Any:
        return self.h[0] if self.h else None
    
    def __len__(self) -> int:
        return len(self.h)
    
    def __str__(self) -> str:
        res = ''
        aux = PriorityQueue()
        
        while not self.empty():
            e = self.pop()
            res += str(e) + '\n'
            aux.push(e)
            
        self.h = aux.h
        return res.strip()
