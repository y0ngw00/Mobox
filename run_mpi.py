from mpi4py import MPI
import numpy as np
import env
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

envx = env.env(rank)

def gather_all(x):
	is_array = isinstance(x, np.ndarray)
	x_buf = np.array([x])
	buffer = np.zeros_like(x_buf)
	buffer = np.repeat(buffer, comm.Get_size(), axis=0)
	MPI.COMM_WORLD.Allgather(x_buf, buffer)
	buffer = list(buffer)
	return buffer

print(f'{rank} {gather_all(envx.generate())}')

