import torch
from torch import svd

def batch_procrustes(X, Y):
    """
        Reference: https://ieeexplore.ieee.org/document/88573
        X: source points (BxNx3 torch tensor)
        Y: target points (BxNx3 torch tensor)
    """
    mux = X.mean(1, keepdim=True)
    muy = Y.mean(1, keepdim=True)

    Sxy = (Y - muy).transpose(1, 2).bmm(X - mux) / X.shape[1]
    U, D, V = svd(Sxy)
    S = torch.eye(3, device=Sxy.device).unsqueeze(0).repeat(X.shape[0], 1, 1)

    S[U.det() * V.det() < 0, -1, -1] = -1

    R = U.bmm(S.bmm(V.transpose(1, 2)))
    t = muy.squeeze() - R.bmm(mux.transpose(1, 2)).squeeze()
    return R, t
    

def procrustes(X, Y):
    """
        Reference: https://ieeexplore.ieee.org/document/88573
        X: source points (Nx3 torch tensor)
        Y: target points (Nx3 torch tensor)
    """
    assert len(X) == len(Y)

    mux = X.mean(0, keepdim=True)
    muy = Y.mean(0, keepdim=True)

    Sxy = (Y - muy).t().mm(X - mux) / len(X)
    U, D, V = Sxy.svd()
    S = torch.eye(3, device=Sxy.device)
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t()))
    t = muy.squeeze() - R.mm(mux.t()).squeeze()
    return R, t


def weighted_procrustes(X, Y, w, eps=torch.finfo(torch.float32).eps):
    """
        Reference: https://ieeexplore.ieee.org/document/88573
        X: source points (Nx3 torch tensor)
        Y: target points (Nx3 torch tensor)
        w: weights of the correspondence (N torch tensor)
    """
    assert len(X) == len(Y) == len(w)

    W1 = torch.abs(w).sum()
    w_norm = (w / (W1 + eps)).unsqueeze(1)
    mux = (w_norm * X).sum(0, keepdim=True)
    muy = (w_norm * Y).sum(0, keepdim=True)
  
    Sxy = (Y - muy).t().mm(w_norm * (X - mux))
    U, D, V = Sxy.svd()
    S = torch.eye(3, device=Sxy.device)
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t()))
    t = muy.squeeze() - R.mm(mux.t()).squeeze()
    return R, t
