def lr(x, y, lr=.0001, ep=.001, max_iter = 10000):
    num_iter = 0
    n = x.shape[0]
    betas = np.zeros(x.shape[1])
    betas = betas + lr * sum([(y[i] - np.dot(betas, x[i])) for i in range(n)])
    sse = sum([y[i] - np.dot(x[i], betas)**2 for i in range(n)])
    for iteration in xrange(max_iter):
        betas = betas + lr * sum([(y[i] - np.dot(betas, x[i])) for i in range(n)])
        e = sum([(y[i] - np.dot(betas, x[i]))**2 for i in range(n)])
        if abs(sse-e) < ep and iteration > 2:
            return betas
        sse = e
    print 'Did not converge'
    return