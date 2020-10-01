import subprocess
import shlex
import collections

PortsConfig = collections.namedtuple('PortsConfig', 'carla waymo others')

# Kills the pid holding a port
def killPrevFlaskInstance(machinePort):
    # Kill previous commands
    cmdFindPidsOnPort = f"lsof -t -i:{machinePort}"
    cmdFindPidsOnPort = shlex.split(cmdFindPidsOnPort)
    res = subprocess.run(cmdFindPidsOnPort, stdout=subprocess.PIPE)
    res = res.stdout.strip().decode('ascii')
    if len(res) > 0:
        pidHoldingPort = int(res)

        cmdKill = f"kill -9 {pidHoldingPort}"
        cmdArgs = shlex.split(cmdKill)
        res = subprocess.run(cmdArgs, stdout=subprocess.PIPE)
        print(res.stdout)

def runFlask(args, app):
    # Choose machine address and port
    machineAddress = "localhost"
    machinePort = args.portsConfig.others
    if "carla" in args.model_path:
        machinePort = args.portsConfig.carla
    elif "waymo" in args.model_path:
        machinePort = args.portsConfig.waymo

    # close existing port to be sure tha we don;t fail
    print("Running deployment for others...")
    app.run(host=machineAddress, port=machinePort, debug=False)
