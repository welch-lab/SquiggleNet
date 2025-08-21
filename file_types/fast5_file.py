
# This is an implementation of the interfaces in signal_file.py for the Fast5 file type.

from typing import override, Iterator
from ont_fast5_api.fast5_interface import get_fast5_file, Fast5File as F5File
from ont_fast5_api.fast5_read import Fast5Read as F5Read
from file_types.signal_file import SignalFile, SignalRead

class Fast5Read(SignalRead):
    '''
    Implementation of SignalRead interface for Fast5 Read/File objects.
    '''
    @override
    def __init__(self, read: F5File | F5Read):
        self.read = read
    
    @override
    def get_read_id(self):
        return self.read.read_id
    
    @override
    def get_raw_signal_pA(self):
        return self.read.get_raw_data(scale=True)


class Fast5File(SignalFile):
    '''
    Implemention of SignalFile interface for Fast5 files.
    '''
    @override
    def __init__(self, filename: str):
        self.f5File = get_fast5_file(filename, mode='r')

    @override
    def __enter__(self):
        return self.f5File.__enter__()
    
    @override
    def __exit__(self, exc_type, exc_value, traceback):
        return self.f5File.__exit__(exc_type, exc_value, traceback)
    
    @override
    def get_reads(self) -> Iterator[SignalRead]:
        return map(Fast5Read, self.f5File.get_reads())