import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

class Paths:

    def __init__(self, num_cells, T):
        self.num_cells = num_cells  # how many rows and cols does our grid map have
        self.T = T # we assume the animation runs for T frames before it stops

    def circle(self, M, x_center, y_center, R, theta_init, spd, t):
        X = []
        Y = []
        omega = spd/R
        theta = theta_init + omega*t # angle theta when the angular speed is set to be a constant
        i = int(y_center) + int(R*np.sin(theta)) # the col position of the vehicle
        j = int(x_center) + int(R*np.cos(theta)) # the row position of the vehicle
        Theta = np.linspace(0, 2*np.pi, 1000)
        for elem in Theta:
            X.append(x_center + R*np.cos(elem)) # a continuous solid curve for the path
            Y.append(y_center + R*np.sin(elem))
        M[i, j, t] = 1
        return (M, X, Y, i, j)

    def line(self, M, x_init, y_init, v_x, v_y, t):
        X = []
        Y = []
        i = int(y_init) + int(v_y*t)
        j = int(x_init) + int(v_x*t)
        for elem in np.linspace(0, self.T, 1000):
            x_new = x_init + v_x*elem
            y_new = y_init + v_y*elem
            if x_new <= self.num_cells and x_new >=0 and y_new <= self.num_cells and y_new >= 0:
                X.append(x_new)
                Y.append(y_new)
        if i < self.num_cells and i >= 0 and j < self.num_cells and j >=0:
            M[i, j, t] = 1
        return (M, X, Y, i, j) 

# A function to visualize what is happening

def visualize(Cells, T, P):
    M = np.zeros((Cells, Cells, T))
    for k in range(0, T):
        t = k
        M, X_1, Y_1, i_1, j_1 = P.circle(M, 50, 50, 20, 0, 5, t)
        M, X_2, Y_2, i_2, j_2 = P.line(M, 0, 0, 1, 0.5, t)
        M, X_3, Y_3, i_3, j_3 = P.line(M, 20, 99, -0.2, -1.25, t)
        M, X_4, Y_4, i_4, j_4 = P.circle(M, 70, 70, 10, 0, -3, t)
        M, X_5, Y_5, i_5, j_5 = P.circle(M, 80, 20, 10, 1.7, 3, t)

    # Animate everything

    fig = plt.figure()
    im = plt.imshow(M[:,:,0], interpolation="none", cmap="Reds")
    title = plt.title("")

    def update(t):
        im.set_array(M[:,:,t])
        title.set_text("Time = " + str(t) + " s")

    ani = matplotlib.animation.FuncAnimation(fig, func=update, frames=T, repeat=False, interval=400)
                       
    plt.plot(X_1, Y_1)
    plt.plot(X_2, Y_2)
    plt.plot(X_3, Y_3)
    plt.plot(X_4, Y_4)
    plt.plot(X_5, Y_5)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

# Testing

Cells = 100
T = 100
P = Paths(Cells, T)
visualize(Cells, T, P)