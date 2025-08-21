
# These are interfaces to future-proof the preprocess step of the model in case new file types come out for Nanopore signal reads in the future.

from typing import Any, Iterator

class SignalRead:
    '''
    Interface for Nanopore signal reads.

    Implement it for whatever formats are used either now or in the future.
    '''
    def __init__(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self

    def get_read_id(self):
        pass

    def get_raw_signal_pA(self):
        pass
    

class SignalFile:
    '''
    Interface for Nanopore signal containing files.

    Implement it for whatever file types are used either now or in the future.

    Must implement __enter__, __exit__ methods to use in Python 'with' expressions.
    '''
    def __init__(self):
        pass
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self
    
    def get_reads(self) -> Iterator[SignalRead]:
        pass
