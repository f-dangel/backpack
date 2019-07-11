

def variance_from(g, sgs, N):
    avgg_squared = (g / N)**2
    avg_gsquared = sgs / N
    return avg_gsquared - avgg_squared
