
# This is an implementation of the interfaces in signal_file.py for the Pod5 file type.

from typing import override, Iterator
from pod5 import ReadRecord, Reader
from file_types.signal_file import SignalFile, SignalRead

class Pod5Read(SignalRead):
    '''
    Implementation of SignalRead interface for Pod5 ReadRecord objects.
    '''
    @override
    def __init__(self, read: ReadRecord):
        self.read = read
    
    @override
    def get_read_id(self):
        return str(self.read.read_id)
    
    @override
    def get_raw_signal_pA(self):
        return self.read.signal_pa
    

class Pod5File(SignalFile):
    '''
    Implementation of SignalFile interface for Pod5 files.
    '''
    @override
    def __init__(self, filename: str):
        self.reader = Reader(filename)

    @override
    def __enter__(self):
        self.reader.__enter__()
        return self

    @override    
    def __exit__(self, exc_type, exc_value, traceback):
        return self.reader.__exit__(exc_type, exc_value, traceback)

    @override
    def get_reads(self) -> Iterator[SignalRead]:
        return map(Pod5Read, self.reader.reads())