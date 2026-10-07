"""Check the eigenvalues of Q_1 from TutorialGO Question 1 with numpy.

Run:  conda run -n optimization python check_eig_q1.py
"""
import numpy as np

Q1 = np.array([[-2, -1, 0, 0],
               [-1, 2, 0, 0],
               [0, 0, 3, -1],
               [0, 0, -1, 2]], float)

ev = np.linalg.eigvalsh(Q1)  # symmetric matrix: real eigenvalues, ascending
print("eig(Q1)         =", np.round(ev, 4))

# Q1 is block diagonal, so its spectrum is the union of the two 2x2 blocks
A, B = Q1[:2, :2], Q1[2:, 2:]
print("eig(A), x1,x2   =", np.round(np.linalg.eigvalsh(A), 4), " (expected +-sqrt(5))")
print("eig(B), x3,x4   =", np.round(np.linalg.eigvalsh(B), 4), " (expected (5 +- sqrt(5))/2)")

# convex split: Q1 + 5/2 e1 e1^T should be PSD and singular
S = Q1 + np.diag([2.5, 0, 0, 0])
print("eig(Q1 + 5/2 E11) =", np.round(np.linalg.eigvalsh(S), 4))

assert np.allclose(ev, np.sort(np.r_[-np.sqrt(5), np.sqrt(5), (5 - np.sqrt(5)) / 2, (5 + np.sqrt(5)) / 2]))
assert abs(np.linalg.eigvalsh(S).min()) < 1e-12
print("all checks passed")
