def lr_poly(base_lr, iter, max_iter, wait_iter=0, power=0.9):
    if iter < max_iter and iter>=wait_iter:
        return base_lr*((1-float(iter-wait_iter)/(max_iter-wait_iter))**(power))
    else:
        return base_lr

def reduce_lr_poly(base_lr, optimizer, global_iter, max_iter, wait_iter=0):
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, wait_iter=wait_iter, power=0.9)