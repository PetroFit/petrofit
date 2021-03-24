import os
from sys import argv

def mount_bind_command(src, dst):
    return "--mount type=bind,src={},dst={}".format(src, dst)

if __name__ == '__main__':
    if len(argv) < 2:
        print('Command not supplied')
        print('python docker.py build')
        print('python docker.py run [extra_mount_paths]')
        print('python docker.py run_safe # (does not mount the host copy of petrofit dir)')

    # Build
    # -----
    # Command to build the Dockerfile
    elif argv[1] == 'build' or argv[1].replace("-", "") == "b":
        os.system('docker build -t petrofit .')

    # Mount and Run
    # -------------
    # Command to run the docker image in a container.
    # This will mount the host version of the repo so any changes to the code
    # in the host repo will be available in the docker container and the environment.
    # Thus the docker container will use the host's version of petrofit module since the
    # environment in the container is installed using `python setup.py develop`.
    # (you are able to update code on host without having to reinstall in the container)
    elif argv[1] == 'run' or argv[1].replace("-", "") == "r":
        path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
        command = "docker run"
        command += " " + mount_bind_command(path_to_this_dir, "/home/jovyan/petrofit")

        # This section mounts (bind) any number of volumes on the host into the docker container.
        # The volumes are mounted into the `mount` folder inside the container.
        # Any edits there will also apply to the host version of the files.
        mounted_list = []
        if len(argv) > 2:
            for path in argv[2:]:
                src = os.path.abspath(path)
                dst = "/home/jovyan/mount/{}".format(os.path.basename(path))
                if dst in mounted_list:
                    raise Exception("Mount request includes multiple files or folders with the same name")
                mounted_list.append(dst)
                command += " " + mount_bind_command(src, dst)

        command += " " + "-it -p 8888:8888 petrofit"

        print(">>>", command, "\n")

        os.system(command)

    # Regular Run
    # -----------
    # Nothing is mounted on to the container.
    elif argv[1] == 'run_safe':
        os.system('docker run -it -p 8888:8888 petrofit')
