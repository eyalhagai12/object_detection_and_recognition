from enum import Enum


class TrackerAlgo(Enum):
    MOSSE = 1
    CSRT = 2
    KCF = 3
    TLD = 4
    MIL = 5
    GOTURN = 6
