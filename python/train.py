import sys
import subprocess
import mpi_train

def main():
	# Command line arguments
	args = sys.argv[1:]
	arg_parser = ArgParser()
	arg_parser.load_args(args)

	Logger.print('Running with 16 workers')
	cmd = 'mpiexec -n 16 python3 mpi_train.py config --config.py'
	Logger.print('cmd: ' + cmd)
	subprocess.call(cmd, shell=True)

if __name__ == '__main__':
	main()
