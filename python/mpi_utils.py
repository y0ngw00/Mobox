from mpi4py import MPI


def get_num_procs():
	return MPI.COMM_WORLD.Get_size()

def get_proc_rank():
	return MPI.COMM_WORLD.Get_rank()

def is_root_proc():
	rank = get_proc_rank()
	return rank == 0

def broadcast(x):
	return MPI.COMM_WORLD.bcast(x, root=0)

def gather(x):
	return MPI.COMM_WORLD.gather(x, root=0)

def scatter(x):
	return MPI.COMM_WORLD.scatter(x, root=0)

