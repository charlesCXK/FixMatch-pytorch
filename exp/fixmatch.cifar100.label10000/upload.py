import os
import sys
import pprint
import argparse
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import ContainerRegistry
from azureml.train.dnn import PyTorch


from azureml.contrib.core.k8srunconfig import K8sComputeConfiguration

from azureml.train.estimator import Estimator
from azureml.core.runconfig import MpiConfiguration

'''
Choose training cluster by use different configs
'''
# import config_ussc as config
subscription_id = "d4c59fc3-1b01-4872-8981-3ee8cbd79d68"
resource_group  = "usscv100"
workspace_name  = "usscv100ws"
cluster_name    = "usscv100cl"

parser = argparse.ArgumentParser()
parser.add_argument('--itp', default=0, type=int)
parser.add_argument('--card', default=4, type=int)

myargs = parser.parse_args()
if myargs.itp == 1:     # usscv100cl, 64 GPUs per user
    subscription_id = "46da6261-2167-4e71-8b0d-f4a45215ce61"
    resource_group = "researchvc-sc"
    workspace_name = "resrchvc-sc"
    cluster_name = "itpscusv100cl"
    nick_name = 'sc'
if myargs.itp == 2:     # philly rr1, 64 GPUs per user
    subscription_id = "46da6261-2167-4e71-8b0d-f4a45215ce61"
    resource_group = "researchvc"
    workspace_name = "resrchvc"
    cluster_name = "itplabrr1cl1"
    nick_name = 'rr1'
if myargs.itp == 3:     # Researchvc-eus, 32 GPUs per user
    subscription_id = "46da6261-2167-4e71-8b0d-f4a45215ce61"
    resource_group = "Researchvc-eus"
    workspace_name = "resrchvc-eus"
    cluster_name = "itpeastusv100cl"
if myargs.itp == 4:     # east asia, 16 GPUs per user
    subscription_id = "46da6261-2167-4e71-8b0d-f4a45215ce61"
    resource_group = "researchvc-sea"
    workspace_name = "resrchvc-sea"
    cluster_name = "itpseasiav100cl"

blob_container_name = "xiaokc"
blob_account_name   = "cxk"
blob_account_key    = "YbLOkA3pNqJUWs5W/5R6D0B3dLkGFYNcRL+1KxbBxw/gPf5ZOrcMYDuxio39et8+0tHgWWsGSw5jdUJFbuwyMQ=="
datastore_name = 'cxk_datastore'

def prepare():
    ws = None
    try:
        print("Connecting to workspace '%s'..." % workspace_name)
        ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
    except:
        print("Workspace not accessible.")
    print(ws.get_details())

    ws.write_config()

    #
    # Register an existing datastore to the workspace.
    #
    if datastore_name not in ws.datastores:
        Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name=datastore_name,
            container_name=blob_container_name,
            account_name=blob_account_name,
            account_key=blob_account_key
        )
        print("Datastore '%s' registered." % datastore_name)
    else:
        print("Datastore '%s' has already been regsitered." % datastore_name)

if __name__ == '__main__':
    # prepare()

    '''
    Submit a job by execute:
      python run_docker_inst.py --cfg xxx
    The xxx will be passed into the 'entry_script', and will also be displayed as the 'tag'
    '''
    parser = argparse.ArgumentParser(description="AML Generic Launcher")
    parser.add_argument("--cfg", default="")
    args, _ = parser.parse_known_args()

    # docker image registry, no need to change if you want to use Philly docker
    container_registry_address = "phillyregistry.azurecr.io/" # example : "phillyregistry.azurecr.io"
    # custom_docker_image ="philly/jobs/custom/pytorch:v1.1-py36-hrnet" # example: "philly/jobs/custom/pytorch:your tag"

    '''
    AML can use Docker images in the DockerHub, specify your Docker image here
    '''
    custom_docker_image = "charlescxk/ssc:pt-1.4" # example: "philly/jobs/custom/pytorch:your tag" "pytorch/pytorch:1.5-cuda10.1-cudnn7-devel"


    '''
    entry_script: Specify the script you want to execute, here I set to be ./docker/inst_efficienthrnet.py as default script
    '''
    # Note: source_directory and entry_script are in local, source_directory/entry_script
    source_directory = "./"
    # print(sys.argv[1])
    # entry_script = sys.argv[1]
    entry_script = 'run.py'
    # entry_script = "./entry-script.py"

    # subscription_id = config.subscription_id
    # resource_group = config.resource_group
    # workspace_name = config.workspace_name
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)

    # cluster_name= config.cluster_name
    ct = ComputeTarget(workspace=ws, name=cluster_name)
    # datastore_name =config.datastore_name
    ds = Datastore(workspace=ws, name=datastore_name)

    workdir = os.path.realpath('.')[os.path.realpath('.').find('FixMatch-pytorch'):]
    workdir = workdir.replace('\\', '/')

    script_params = {
        "--workdir": ds.path('/projects/'+workdir).as_mount(), # REQUIRED !!!
        "--cxk_volna": ds.path('/').as_mount(),
        "--exp_name": workdir.split('/')[-1],
    }

    def make_container_registry(address, username, password):
        cr = ContainerRegistry()
        cr.address = address
        cr.username = username
        cr.password = password
        return cr


    estimator = PyTorch(source_directory='./',
                        script_params=script_params,
                        compute_target=ct,
                        use_gpu=True,
                        shm_size='256G',
                        # image_registry_details= my_registry,
                        entry_script=entry_script,
                        custom_docker_image=custom_docker_image,
                        user_managed=True,
                        )


    if myargs.itp > 0:
        cmk8sconfig = K8sComputeConfiguration()

        cmk8s = dict()
        cmk8s['gpu_count'] = myargs.card

        cmk8sconfig.configuration = cmk8s
        estimator.run_config.cmk8scompute = cmk8sconfig

    experiment = Experiment(ws, name='semiexp')

    run = experiment.submit(estimator, tags={'tag': workdir.split('/')[-1]})

    pprint.pprint(run)

    # uncomment next line to see the stdout in your main.py on local machine.
    #run.wait_for_completion(show_output=True)

    # (END)
