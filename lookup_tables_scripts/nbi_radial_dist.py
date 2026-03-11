import numpy as np
import matplotlib.pyplot as plt
import sys

class radial_weight:
    def __init__(self, N, y_vals):
        self.N = N
        self.r_max = np.max(y_vals)
        self.r_vals = y_vals[y_vals > -1e-10]
        self.len = len(y_vals)
        if (self.len % 2 == 0 and N % 2 != 0) or (self.len % 2 !=0 and N % 2 == 0):
            print('Beamlet grid and radial count must be both odd or even to preserve rotational symmetry!')
            sys.exit()

    def uniform_grid(self, N, r_max):
        x = np.linspace(-r_max, r_max, N)
        y = np.linspace(-r_max, r_max, N)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def hex_grid(self, N, r_max):
        x = np.linspace(-r_max, r_max, N)
        y = np.linspace(-r_max, r_max, N)
        xx, yy = np.meshgrid(x, y)
        ratio = x[1] - x[0]
        xx[::2,:] += ratio / 2
        return np.column_stack((xx.ravel(), yy.ravel()))

    def dist(self, points):
        return np.linalg.norm(points, axis=1)

    def counter(self, cart=False, plots=0):
        theta = np.linspace(0, 2 * np.pi, 100)
        radii = self.r_vals
        if cart == True:
            points = self.uniform_grid(self.N, np.max(radii))
        else:
            points = self.hex_grid(self.N, np.max(radii))
        dr = np.diff(radii)
        dr[0] = dr[0] - 1e-10
        norm = self.dist(points)

        count = np.zeros_like(radii)

        if plots == 1:
            fig = plt.figure(figsize=(8,8))
            plt.plot(points[:,0], points[:,1], '.', c='r', label='beamlet')
        
        for i, r in enumerate(radii):
            if i == 0:
                count[i] = np.sum(norm <= (r + 1e-10))
            else:
                count[i] = np.sum((norm > r-dr[i-1]) & (norm <= r))
            if plots == 1:
                plt.plot(r*np.cos(theta), r*np.sin(theta), c='k')
                print(f'There are {count[i]} beamlets when r = {r}')

        if plots == 1:
            plt.title('Beamlet Grid')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.axis('equal')
            plt.tight_layout()
            plt.close()
            # plt.show()

        if self.len % 2 == 0:
            count = np.concatenate((np.flip(count / 2), count / 2))
        else:
            count = np.concatenate((np.flip(count[1:] / 2), [count[0]], count[1:] / 2))

        return np.round(count).astype(np.int64)

