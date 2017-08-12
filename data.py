import matplotlib.pylab as plt
import numpy as np
import copy

# import matplotlib.pyplot as plt

np.random.seed(3)

class Y_dataset(object):  # y_data製造用class

    def __init__(self, x_data): # n:scalar x:
        self.x = x_data
        self.y_data = self.make_data()

    def make_data(self):  #input x:x_data(n-dim)// output y_data(np.array,n-dim
        dataplot = []
        for i in self.x:
            tmp = 2 * i
            tmp += 0.5 *i*np.random.randn()
            dataplot.append(tmp)

        dataplot = np.array(dataplot)
        return dataplot


class Agent(object):
    def __init__(self, x_data,n):#x_data np.array n-dim
        self.x_i = np.ones(n)
        self.x_data = x_data

        self.n = n
        self.eta = 0.01
        self.phi = copy.copy(self.make_phi(x_data))
        self.phi_to = copy.copy(self.phi.T)
        self.phi_tophi = copy.copy(np.dot(self.phi_to,self.phi))
        # self.x_i = self.optimal()

    def make_phi(self, x):
        phi = np.array([[(i ** (j )) for j in range(self.n)] for i in x_data])
        return phi


    def get_y_data(self,y_data):#input y_data(np.array,n-dim)
        self.y_data = y_data.T

    def update(self):
        self.x_i = copy.copy(self.x_i - self.eta * (np.dot(self.phi_tophi,self.x_i) - np.dot(self.phi_to,self.y_data)))
        # self.x_i = self.projection0_to1(self.x_i)
        # self.x_i
        # print((np.dot(self.phi_tophi,self.x_i) - np.dot(self.phi_to,self.y_data)))
        # print(self.x_i)
        print(np.linalg.norm(np.dot(self.phi_tophi,self.x_i) - np.dot(self.phi_to,self.y_data)))
        # print(np.linalg.norm(self.x_opt-self.x_i))


    def projection0_to1(self,x):
        tmp = x
        for i in range(len(x)):
            if tmp[i] < -1:
                tmp[i] = -1
            elif tmp[i] >1:
                tmp[i] = 1

        return tmp



    def optimal(self):
        invphi_tophi = np.linalg.inv(self.phi_tophi)
        self.x_i = np.dot(np.dot(invphi_tophi,self.phi_to),self.y_data)

        self.x_opt = np.dot(np.dot(invphi_tophi,self.phi_to),self.y_data)
    def write_graph(self):
        graph_x_tick =  np.array([0.001 * (i) for i in range(100*len(self.x_data))])
        graph_x_tick_ydata =  np.array([0.1 * (i) for i in range(len(self.x_data))])

        phi = np.array([[(i ** (j)) for j in range(n)] for i in graph_x_tick])
        graph_data = np.dot(phi,self.x_i)
        # print(graph_data)
        plt.plot(graph_x_tick,graph_data)
        plt.plot(graph_x_tick_ydata ,self.y_data,'o')
        plt.show()


if __name__ == '__main__':
    x_data = np.array([0.1 * (i) for i in range(10)])
    n = len(x_data)
    agent = Agent(x_data,n)
    data = Y_dataset(x_data)
    agent.get_y_data(data.y_data)

    agent.optimal()
    for k in range(1000000):
        agent.update()

    # agent.optimal()
    agent.write_graph()


    # print(data.data)
    # x = np.arange(0, 10)
    # plt.plot(x, data.data, 'o')
    # plt.show()
