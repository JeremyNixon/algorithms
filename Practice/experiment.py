def lr_gd(x, y, lr=.0001, ep=.001, max_iter = 10000):
    n = x.shape[0]
    betas = np.zeros(x.shape[1])
    betas = betas + lr * sum([(y[i] - np.dot(betas, x[i])) for i in range(n)])
    sum_squared_error = sum([(y[i]-np.dot(betas, x[i]))**2 for i in range(n)])
    for iteration in xrange(max_iter):
        betas = betas + lr * sum([(y[i]-np.dot(betas, x[i]))*x[i] for i in range(n)])
        error = sum([(y[i]-np.dot(betas, x[i])**2) for i in range(n)])
        if abs(sum_squared_error-error) < ep:
            return betas
        sum_squared_error = error
    print 'Failed to converge'