import string
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a file generator class for LKH

class LKH_file_generator:

    def __init__(self, C, filename_tsp, filename_par, filename_sol):
        self.C = C
        self.filename_tsp = filename_tsp
        self.filename_par = filename_par
        self.filename_sol = filename_sol

    def compile_row_string(self, a_row):
        return str(a_row).strip(']').strip('[').replace(',','')

    def create_TSP(self, name = 'test'): # here, the user inputs the cities, and their coords into the C matrix.
        with open(self.filename_tsp, 'w') as f:
            f.write('NAME : %s\n' % name)
            f.write('COMMENT : few cities test problem\n')
            f.write('TYPE : ATSP\n')
            f.write('DIMENSION : %d\n' % len(self.C))
            f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
            f.write('NODE_COORD_SECTION\n')
            for row in self.C:
                f.write('%d %d %d\n' % (row[0], row[1], row[2]))
            f.write('EOF\n')

    def create_cost_matrix_TSP(self, name = 'test_1'): # here, the user can input the cost matrix directly.
        with open(self.filename_tsp, 'w') as f:
            f.write('NAME : %s\n' % name)
            f.write('COMMENT : few cities test problem\n')
            f.write('TYPE : ATSP\n')
            f.write('DIMENSION : %d\n' % len(self.C))
            f.write('EDGE_WEIGHT_TYPE : EXPLICIT\n')
            f.write('EDGE_WEIGHT_FORMAT : FULL_MATRIX\n')
            f.write('EDGE_WEIGHT_SECTION\n')
            for row in self.C:
                f.write(self.compile_row_string(row) + '\n')
            f.write('EOF\n')

    def create_PAR(self, name = 'test.tsp', tour = 'testsol'):
        with open(self.filename_par, 'w') as f:
            f.write('PROBLEM_FILE = %s\n' % name)
            f.write('TOUR_FILE = %s\n' % tour)
            f.write('RUNS = 10')

    def create_cost_matrix_PAR(self, name = 'test_1.tsp', tour = 'test_1sol'):
        with open(self.filename_par, 'w') as f:
            f.write('PROBLEM_FILE = %s\n' % name)
            f.write('TOUR_FILE = %s\n' % tour)
            f.write('RUNS = 10')

    def read_sol(self):
        F = []
        with open(self.filename_sol) as f:
            for index, line in enumerate(f):
                if index > 5 and index < len(self.C) + 6:
                    F.append(int(line))
        return F      


class Moving_TSP:

    def __init__(self, num_cells, T, m, V):
        self.num_cells = num_cells  # how many rows and cols does our grid map have
        self.T = T # we assume the animation runs for T frames before it stops
        self.V = V # contains info about the vehicle(s): x,y coordinates of the depot, and the speed
        self.circle_trajectories = [] # all the info for circular trajectories for the problem will be appended within this matrix
        self.line_trajectories = [] # all the info for line trajectories for the problem will be appended within this matrix
        self.m = m # the total number of targets that we have
        self.M = np.zeros((self.num_cells, self.num_cells, self.T)) # the grid map that gets updated every frame
        
    def circle(self, x_center, y_center, R, theta_init, spd, t):
        X = []
        Y = []
        omega = spd/R
        theta = theta_init + omega*t # angle theta when the angular speed is set to be a constant
        x = x_center + R*np.cos(theta) # actual x coordinate of target
        y = y_center + R*np.sin(theta) # actual y coordinate of target
        i = int(y_center) + int(R*np.sin(theta)) # the row position of the target for animation
        j = int(x_center) + int(R*np.cos(theta)) # the col position of the target for animation
        Theta = np.linspace(0, 2*np.pi, 1000)
        for elem in Theta:
            X.append(x_center + R*np.cos(elem)) # a continuous solid curve for the path for animation
            Y.append(y_center + R*np.sin(elem))
        self.M[i, j, t] = 1
        return [x, y, X, Y, i, j]

    def line(self, x_init, y_init, v_x, v_y, t):
        X = []
        Y = []
        x = x_init + v_x*t # actual x coordinate of target
        y = y_init + v_y*t # actual y coordinate of target
        i = int(y_init) + int(v_y*t) # the row position of the target for animation
        j = int(x_init) + int(v_x*t) # the col position of the target for animation
        for elem in np.linspace(0, self.T, 1000):
            x_new = x_init + v_x*elem
            y_new = y_init + v_y*elem
            if x_new <= self.num_cells and x_new >=0 and y_new <= self.num_cells and y_new >= 0:
                X.append(x_new) # a continuous solid curve for the path for animation
                Y.append(y_new)
        if i < self.num_cells and i >= 0 and j < self.num_cells and j >=0:
            self.M[i, j, t] = 1
        return [x, y, X, Y, i, j] 

    def add_circle_trajectories(self, x_center, y_center, R, theta_init, spd): # user can add circle trajectories
        self.circle_trajectories.append([x_center, y_center, R, theta_init, spd])

    def add_line_trajectories(self, x_init, y_init, v_x, v_y): # user can add line trajectories
        self.line_trajectories.append([x_init, y_init, v_x, v_y])

    def create_coord_matrix(self):
        coord_matrix = []
        for i in range(len(self.circle_trajectories)):
            coord_matrix.append([])
            cr_in = self.circle_trajectories[i]
            for k in range(self.T):
                t = k
                cr = [self.circle(cr_in[0], cr_in[1], cr_in[2], cr_in[3], cr_in[4], t)[0], 
                self.circle(cr_in[0], cr_in[1], cr_in[2], cr_in[3], cr_in[4], t)[1]]
                coord_matrix[i].append(cr)
        for i in range(len(self.circle_trajectories), len(self.circle_trajectories) + len(self.line_trajectories)):
            coord_matrix.append([])
            ln_in = self.line_trajectories[i - len(self.circle_trajectories)]
            for k in range(self.T):
                t = k
                ln = [self.line(ln_in[0], ln_in[1], ln_in[2], ln_in[3], t)[0], 
                self.line(ln_in[0], ln_in[1], ln_in[2], ln_in[3], t)[1]]
                coord_matrix[i].append(ln)
        return coord_matrix

    def plot_trajectories(self):
        plot_matrix_X = []
        plot_matrix_Y = []
        for i in range(len(self.circle_trajectories)):
            cr_in = self.circle_trajectories[i]
            plot_matrix_X.append(self.circle(cr_in[0], cr_in[1], cr_in[2], cr_in[3], cr_in[4], 0)[2])
            plot_matrix_Y.append(self.circle(cr_in[0], cr_in[1], cr_in[2], cr_in[3], cr_in[4], 0)[3])
        for i in range(len(self.line_trajectories)):
            ln_in = self.line_trajectories[i]
            plot_matrix_X.append(self.line(ln_in[0], ln_in[1], ln_in[2], ln_in[3], 0)[2])
            plot_matrix_Y.append(self.line(ln_in[0], ln_in[1], ln_in[2], ln_in[3], 0)[3])
        return [plot_matrix_X, plot_matrix_Y]

    def visualize(self):
        # Animate everything
        fig = plt.figure()
        im = plt.imshow(self.M[:,:,0], interpolation="none", cmap="Reds")
        title = plt.title("")
        def update(t):
            im.set_array(self.M[:,:,t])
            title.set_text("Time = " + str(t) + " s")
        ani = matplotlib.animation.FuncAnimation(fig, func=update, frames=self.T, repeat=False, interval=400)
        # Plot the circle and line trajectories the object moves over
        for i in range(len(self.plot_trajectories()[0])):
            plt.plot(self.plot_trajectories()[0][i], self.plot_trajectories()[1][i]) 
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        # Save the animation in MP4 format
        f = r"/home/nykabhishek/George_Allen/Research_slides/mar_11_2022.mp4" 
        writervideo = matplotlib.animation.FFMpegWriter(fps=3) 
        ani.save(f, writer=writervideo)

    def distance(self, x_1, y_1, x_2, y_2):
        d = ((x_2 - x_1)**2 + (y_2 - y_1)**2)**0.5
        return d

    def problem_graph_1(self): # define the first part of the problem as a graph
        graph = []
        max = 141
        M = 2*max*(self.m + self.T) # an approximate upperbound on M
        coord = self.create_coord_matrix()

        for row in range(1 + self.T*self.m):
            graph.append([])
            for col in range(1 + self.T*self.m):
                graph[row].append(9999999)

        for i in range(self.m): # check if edges can be created between sets. If true, make the edge as per Noon Bean.
            for j in range(self.T):
                for k in range(self.m):
                    for l in range(j, self.T): # j + 1 is faster, but targets might occupy the same position. Hence j
                        if k != i:
                            if j != 0:
                                d_1 = self.distance(coord[i][j][0], coord[i][j][1], coord[k][l][0], coord[k][l][1])
                                if d_1/self.V[0][2] <= l - j:
                                    graph[1 + (i)*self.T + (j - 1)][1 + (k)*self.T + l] = d_1 + M
                            elif j == 0:
                                d_2 = self.distance(coord[i][j][0], coord[i][j][1], coord[k][l][0], coord[k][l][1])
                                if d_2/self.V[0][2] <= l - j:
                                    graph[1 + (i)*self.T + (self.T - 1)][1 + (k)*self.T + l] = d_2 + M
        
        for i in range(self.m): # make a directed cycle of cost 0 within each set as per Noon Bean.
            for j in range(self.T):
                if j != self.T - 1:
                    graph[1 + (i)*self.T + j][1 + (i)*self.T + (j + 1)] = 0
                elif j == self.T - 1:
                    graph[1 + (i)*self.T + j][1 + (i)*self.T + 0] = 0

        for i in range(self.m):
            for j in range(self.T):
                d_3 = self.distance(self.V[0][0], self.V[0][1], coord[i][j][0], coord[i][j][1])
                if d_3/self.V[0][2] <= j:
                    graph[0][1 + (i)*self.T + j] = d_3 + M # for now assume, vehicle to target and target to vehicle is same cost
                graph[1 + (i)*self.T + j][0] = d_3 + M
        
        for i in range(len(graph)): # change the input entries to integer form for LKH
            for j in range(len(graph)):
                graph[i][j] = int(graph[i][j])
        
        return graph

    def problem_feasible_sol_1(self, F): # Extract the feasible sol to one in a set from the LKH sol for ATSP
        P = []
        for i in range(self.m):
            k = 1 + i*self.T
            j = (F[k] - 1)/self.T
            if j - int(j) == 0:
                j = int(j) - 1
            else:
                j = int(j)
            l = (F[k] - 1) - j*self.T 
            P.append([F[k], j, l])
        return P

    def plot_feasible_sol_1(self, F, coord_matrix): # plot the feasible sol to one in a set TSP
        Plot = self.problem_feasible_sol_1(F)
        Plot_X = []
        Plot_Y = []
        Plot_X_Scatter = []
        Plot_Y_Scatter = []
        Plot_Matrix = self.plot_trajectories()
        for i in range(len(Plot_Matrix[0])):
            plt.plot(Plot_Matrix[0][i], Plot_Matrix[1][i])
        Plot_X.append(self.V[0][0])
        Plot_Y.append(self.V[0][1])
        for i in range(len(Plot)):
            Plot_X.append(coord_matrix[Plot[i][1]][Plot[i][2] - 1][0])
            Plot_Y.append(coord_matrix[Plot[i][1]][Plot[i][2] - 1][1])
            Plot_X_Scatter.append(coord_matrix[Plot[i][1]][Plot[i][2] - 1][0])
            Plot_Y_Scatter.append(coord_matrix[Plot[i][1]][Plot[i][2] - 1][1])
            print(coord_matrix[Plot[i][1]][Plot[i][2] - 1][0], coord_matrix[Plot[i][1]][Plot[i][2] - 1][1])
        Plot_X.append(self.V[0][0])
        Plot_Y.append(self.V[0][1])
        plt.plot(Plot_X, Plot_Y, 'r')
        plt.scatter(Plot_X_Scatter, Plot_Y_Scatter, c ='r')
        plt.scatter(self.V[0][0], self.V[0][1], c = 'b')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Feasible solution to Moving TSP')
        plt.show()


        


# Test everything out here

V = [[10, 10, 300]]
P = Moving_TSP(100, 100, 5, V)
P.add_circle_trajectories(70, 70, 10, 0, -3)
P.add_circle_trajectories(80, 20, 10, 1.7, 3)
P.add_circle_trajectories(50, 50, 20, 0, 5)
P.add_line_trajectories(0, 0, 1, 0.5)
P.add_line_trajectories(20, 99, -0.2, -1.25)
coord_matrix = P.create_coord_matrix()  
P.visualize()
graph_1 = P.problem_graph_1()

LKH_1 = LKH_file_generator(graph_1, '/home/nykabhishek/George_Allen/LKH/LKH-2.0.9/test_1.tsp', 
'/home/nykabhishek/George_Allen/LKH/LKH-2.0.9/test_1.par', '/home/nykabhishek/George_Allen/LKH/LKH-2.0.9/test_1sol')
LKH_1.create_cost_matrix_TSP()
LKH_1.create_cost_matrix_PAR()
F = LKH_1.read_sol()

P.plot_feasible_sol_1(F, coord_matrix)


