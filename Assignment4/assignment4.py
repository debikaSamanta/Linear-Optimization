"""
Nisha M - CS19BTECH11012
Siddhant Chandorkar - MA19BTECH11003
"""


"""
The input to the code is provided in a text file. The text file is filled with the test case mentioned in the assignment.
1. The first line indicates the size of the matrix A in the form m n.
2. The next m lines indicate the rows of the matrix with values .
3. next line indicates the vector b.
4. The last line is for the cost vector.
6. The assumptions include :  rank of A = n
7. initial feasible point is not given
"""
import numpy as np

epsilon = 1e-8
class simplex_algorithm:
    def __init__(self,A,b,c):
        self.A = A
        self.b = b
        self.c = c
        size = A.shape
        self.m = size[0]
        self.n = size[1]
        self.b = self.unmask_degeneracy()
        self.z = self.feasible_point()
        #after here we get the feasible point which would be used as uaual in the further functions


        result=self.main_simplex()
        print(f"optimal solution is {result}")
        print(f"The Maximum Value is {np.dot(self.c,result)}")

    #based on the input initialize the random value that would be added to each of the rows
    def initialize_random_val(self,a):
        if(a==epsilon):
            self.rand_val= np.random.uniform(epsilon, epsilon * 10, size=self.row_mod)
        else:
            self.rand_val = np.random.uniform(0.1, 10, size=self.row_mod)

    #function to get the feasible point for the specified configuration
    def feasible_point(self):
        if np.all((self.b>=0)):
            t=self.c.shape
            return np.zeros(t)
        else:
            j=0
            while(j<50):
                rand_ind=np.random.choice(self.m,self.n)
                r_A=self.A[rand_ind]
                r_b=self.b[rand_ind]
                try:
                    temp=np.linalg.inv(r_A)
                    temp1=np.dot(temp,r_b)
                    temp2=np.dot(self.A, temp1) - self.b
                    if(np.all((temp2<=0))):
                        return temp1
                    else:
                        continue
                except:
                    pass
                j=j+1

    #function to make it non - degenerate
    #function runs as long as the points becomes feasible
    def unmask_degeneracy(self):
        self.row_mod=self.m-self.n
        i=0
        num=1000
        while(1):
            if(i>=num):
                X = self.b
                self.initialize_random_val(0.1)
                X[:self.row_mod] = X[:self.row_mod] + self.rand_val

            else:
                i = i + 1
                X = self.b
                self.initialize_random_val(epsilon)
                X[:self.row_mod] = X[:self.row_mod] + self.rand_val

            Y = self.feasible_point()
            e_ind = np.where(np.abs(np.dot(self.A, Y) - self.b) < epsilon)[0]
            #degeneracy is removed if the number of equality indices is equal to the columsn of the matrix
            if len(e_ind) == self.n:
                print("Degeneracy is removed")
                break
        return X

    def direction(self):
        x = np.dot(self.A, self.z) - self.b
        e_ind =  np.where(np.abs(x) < epsilon)[0]
        A_dash = self.A[e_ind]
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
    fptr=open('input.txt')
    m, n = map(int, fptr.readline().split())
    A = np.empty([m, n])
    for i in range(m):
        line = fptr.readline().split()
        for j in range(len(line)):
            A[i, j] = float(line[j])
    b = np.asarray(list(map(float, fptr.readline().split())))
    c = np.asarray(list(map(float, fptr.readline().split())))

    print(f" A : {A} \n B : {b} \n C : {c} \n ")
    print("-------------------------------------------")

    simplex_algorithm(A,b,c)

if __name__ == '__main__':
    main()
