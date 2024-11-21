# Function to generate all subsets
import itertools
from typing import Any


def all_subsets(
    lst:list[Any],
    exclude_empty_set:bool=True
):
    # List to store all subsets 
    subsets = []
    # Loop over all possible subset sizes
    for r in range(len(lst) + 1):
        if exclude_empty_set==False or (exclude_empty_set==True and r!=0):
            # Generate all combinations of size r
            subsets.extend(itertools.combinations(lst, r))
    return subsets
