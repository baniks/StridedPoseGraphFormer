import numpy as np


class Graph():
    """ The Graph to model the skeletons of human body/hand
    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration
        layout (string): must be one of the follow candidates
        - 'hm36_gt' same with ground truth structure of human 3.6 , with 17 joints per frame
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1,
                 with_hip=False):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout, with_hip)
        # joint_id_to_name = {0: 'RHip', 1: 'RKnee', 2: 'RFoot', 3: 'LHip', 4: 'LKnee', 5: 'LFoot', 6: 'Spine', 7: 'Thorax', 8: 'Neck', 9: 'Head', 10: 'LShoulder', 11: 'LElbow', 12: 'LWrist', 13: 'RShoulder', 14: 'RElbow', 15: 'RWrist'}
        # print("Edge\n: ")
        # for i,j in self.edge:
        #     print("\t", joint_id_to_name[i], joint_id_to_name[j])

        self.hop_dis = self.get_hop_distance(self.num_node,
                                             self.edge,
                                             max_hop=max_hop)
        # print("hop_dis:\n", self.hop_dis)
        
        self.dist_center = self.get_distance_to_center(layout, with_hip)
        # print("dist_center:\n")
        # for i in range(len(self.dist_center)):
        #     print("\t", joint_id_to_name[i], self.dist_center[i])

        self.get_adjacency(strategy)

    def __str__(self):
        return self.A


    def get_hop_distance(self, num_node, edge, max_hop=1):
        """
            for self link hop_dis=0
            for one hop neighbor hop_dis=1
            rest hop_dis=inf
        """
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            # bidirectional
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


    def get_distance_to_center(self,layout, with_hip=False):
        """
        :return: get the distance of each node to center
        """

        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            if with_hip == False:
                # center = spine
                dist_center[0 : 6] = [1, 2, 3, 1, 2, 3] # legs
                dist_center[6 : 10] = [0, 1, 2, 3] # body
                dist_center[10 : 16] = [2, 3, 4, 2, 3, 4] #arms
            else:
                # center = hip
                dist_center[0 : 7] = [0, 1, 2, 3, 1, 2, 3] # legs
                dist_center[7 : 11] = [1, 2, 3, 4] # body
                dist_center[11 : 17] = [3, 4, 5, 3, 4, 5] #arms

        return dist_center

    def get_edge(self, layout, with_hip=False):
        """
        get edge link of the graph
        self link +  one hop neighbors
        cb: center bone
        """
        if layout == 'hm36_gt':
            if with_hip == False:
                self.num_node = 16
                self_link = [(i, i) for i in range(self.num_node)]

                neighbour_link = [ (6,0), (0,1), (1,2), # Rleg
                                (6,3), (3,4), (4,5), # Lleg
                                (6,7), (7,8), (8,9), # body
                                (7,10), (10,11), (11,12), # Larm
                                (7,13), (13,14), (14,15) # Rarm
                                ]
            else:
                self.num_node = 17
                self_link = [(i, i) for i in range(self.num_node)]

                neighbour_link = [ (0,1), (1,2), (2,3), # Rleg
                                (0,4), (4,5), (5,6), # Lleg
                                (0,7), (7,8), (8,9), (9,10), # body
                                (8,11), (11,12), (12,13), # Larm
                                (8,14), (14,15), (15,16) # Rarm
                                ]
                
            self.edge = self_link + neighbour_link

            # center node of body/hand
            self.center = 6 # Spine

        else:
            raise ValueError("Do Not Exist This Layout.")


    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        
        # creates self link and connects one hop neighbors
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                #a_sym = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.dist_center[j] == self.dist_center[i]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] > self.dist_center[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:                    
                    A.append(a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")



def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
