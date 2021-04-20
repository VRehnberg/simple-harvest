import numpy as np

def polygon_area(x, y):
    '''Shoelace formula
    
    https://stackoverflow.com/a/30408825/15399131
    '''
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    )

def generalized_gini(income):
    '''Generalised Gini coeff from Rafinetti, Siletti and Vernizzi
       
    https://doi.org/10.1007/s10260-014-0293-4
    '''
    if (income == 0).all():
        # No income to all is equality
        return 0.0

    n = len(income)
    # Calculate generalised Lorenz curve
    x = np.linspace(0, 1, n + 1)
    y = np.zeros(n + 1)
    y[1:] = np.cumsum(np.sort(income))
    y_lower = y.copy()
    y_lower[1:-1] = y.min()
    
    total_area = polygon_area(x, y_lower)
    upper_area = polygon_area(x, y)

    return upper_area / total_area

def regular_gini(income):
    '''Regular Gini coefficient'''
    if (income == 0).all():
        # No income to all is equality
        return 0.0

    n = len(income)
    # Calculate generalised Lorenz curve
    x = np.linspace(0, 1, n + 1)
    y = np.zeros(n + 1)
    y[1:] = np.cumsum(np.sort(income))
    
    total_area = y[-1] / 2
    upper_area = polygon_area(x, y)

    return upper_area / total_area
