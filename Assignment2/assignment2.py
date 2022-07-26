"""
Nisha M - CS19BTECH11012
Siddhant Chandorkar - MA19BTECH11003
"""

"""
The input to the code is provided in a text file. The text file is filled with the test case mentioned in the assignment.
1. The first line indicates the size of the matrix A in the form m n.
2. The next m lines indicate the rows of the matrix with values .
3. next line indicates the vector b.
4. Next line is for the cost vector.
5. The last line is the initial feasible point which is assumed to be given
6. The assumptions include :  non - degenerate , rank of A = n
"""

import numpy as np

epsilon = 1e-8
class simplex_algorithm:
    def __init__(self,A,b,c,z):
        self.A = A
        self.b = b
        self.c = c
        self.z = z
        size=A.shape
        self.m=size[0]
        self.n=size[1]

        result=self.main_simplex()
        print(f"optimal solution is {result}")
        print(f"The Maximum Value is {np.dot(self.c,result)}")

    def direction(self):
        x = np.dot(self.A, self.z) - self.b
        e_ind =  np.where(np.abs(x) < epsilon)[0]
        A_dash = self.A[e_ind]
        print(self.A.shape)
        self.d_vec=-np.linalg.inv(np.transpose(A_dash))

    def maximum_alpha(self):
        n=self.n_b - np.dot(self.n_A,self.z)
        d=np.dot(self.n_A,self.u)
        n=n[np.where(d>0)[0]]
        d = d[np.where(d > 0)[0]]
        s = n / d
        self.alpha=np.min(s[s >= 0])



    def neighbour(self):
        self.direction()
        ct = np.dot(self.d_vec,self.c)
        temp = np.where(ct > 0)
        p_cost = temp[0]
        if len(p_cost)==0:
            return None
        else:
            self.u = self.d_vec[p_cost[0]]
            temp1=np.where(np.dot(self.A,self.u)>0)
            temp2=temp1[0]
            if len(temp2)==0:
                print("Unbounded LP")
                exit()

            x = np.abs(np.dot(self.A, self.z) - self.b)
            e_ind = np.where(x < epsilon)[0]
            n_e_ind = ~np.isin(np.arange(len(self.A)), e_ind)

            self.n_A=self.A[n_e_ind]
            self.n_b=self.b[n_e_ind]

            self.maximum_alpha()

            return self.z+self.alpha * self.u

    def main_simplex(self):
        while True:
            result = self.neighbour()
            if result is None:
                break
            else:
                self.z = result
        return self.z

def main():
    fptr=open('input2.txt')
    m, n = map(int, fptr.readline().split())
    A = np.empty([m, n])
    for i in range(m):
        line = fptr.readline().split()
        for j in range(len(line)):
            A[i, j] = float(line[j])
    b = np.asarray(list(map(float, fptr.readline().split())))
    c = np.asarray(list(map(float, fptr.readline().split())))
    z = np.asarray(list(map(float, fptr.readline().split())))
    print(f" A : {A} \n B : {b} \n C : {c} \n z : {z}")
    print("-------------------------------------------")

    simplex_algorithm(A,b,c,z)

if __name__ == '__main__':
    main()
