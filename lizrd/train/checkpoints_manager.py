import json
from copy import deepcopy
import fcntl
from lizrd.train.load_and_save_model import save_checkpoint
import torch
from datetime import datetime
from typing import Optional, Union
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)

from lizrd.support.logging import JointLogger

EXPERIMENT_CHECKPOINT_MANAGER = "checkpoint_manager.json"

CHECKPOINTS_TAG = "checkpoints"
START_NEW_TAG = "START"

CHECKPOINT_STATUS = "status"
CHECKPOINT_STATUS_RUNNING = "RUNNING"
CHECKPOINT_STATUS_PENDING = "PENDING"
CHECKPOINT_STATUS_FINISHED = "FINISHED"
MODEL_CHECKPOINT = "model_checkpoint"
CHECKPOINT_RUNNING_JOB_ID = "running_job_id"
CHECKPOINT_CREATOR_JOB_ID = "creator_job_id"
CHECKPOINT_START_TIMESTAMP = "start_timestamp"
CHECKPOINT_CREATE_TIMESTAMP = "create_timestamp"
CHECKPOINT_STOP_TIMESTAMP = "stop_timestamp"
FINAL_MODEL_CHECKPOINT = "final_model_checkpoint"
CHECKPOINT_METADATA_TAG = "metadata"
SLIDE_METADATA = "trapezoidal_slide"


class Locker:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__ (self):
        self.fp = open(self.filename, self.mode)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
        return self.fp

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

def manager_start_checkpoint(job_id, timestamp):
    return {
        CHECKPOINT_STATUS:CHECKPOINT_STATUS_RUNNING, 
        MODEL_CHECKPOINT:None,
        CHECKPOINT_RUNNING_JOB_ID:job_id, 
        CHECKPOINT_START_TIMESTAMP:timestamp,
        CHECKPOINT_CREATOR_JOB_ID:job_id, 
        CHECKPOINT_CREATE_TIMESTAMP:timestamp,
        CHECKPOINT_STOP_TIMESTAMP:None,
        FINAL_MODEL_CHECKPOINT:None,
        CHECKPOINT_METADATA_TAG:None
    }

def crate_manager_checkpoint(model_checkpoint_path:str, job_id:str, timestamp:str, metadata:Optional[str]=None ):
    return {
        CHECKPOINT_STATUS:CHECKPOINT_STATUS_PENDING, 
        MODEL_CHECKPOINT:model_checkpoint_path,
        CHECKPOINT_RUNNING_JOB_ID:None, 
        CHECKPOINT_START_TIMESTAMP:None,
        CHECKPOINT_CREATOR_JOB_ID:job_id, 
        CHECKPOINT_CREATE_TIMESTAMP:timestamp,
        CHECKPOINT_STOP_TIMESTAMP:None,
        FINAL_MODEL_CHECKPOINT:None,
        CHECKPOINT_METADATA_TAG:metadata
    }

def run_manager_checkpoint(job_id:str, timestamp:str, checkpoint:dict):
    assert checkpoint[CHECKPOINT_STATUS] == CHECKPOINT_STATUS_PENDING
    return {
            CHECKPOINT_STATUS:CHECKPOINT_STATUS_RUNNING, 
        MODEL_CHECKPOINT:checkpoint[MODEL_CHECKPOINT],
            CHECKPOINT_RUNNING_JOB_ID:job_id, 
            CHECKPOINT_START_TIMESTAMP:timestamp,
        CHECKPOINT_CREATOR_JOB_ID:checkpoint[CHECKPOINT_CREATOR_JOB_ID], 
        CHECKPOINT_CREATE_TIMESTAMP:checkpoint[CHECKPOINT_CREATE_TIMESTAMP],
        CHECKPOINT_STOP_TIMESTAMP:None,
        FINAL_MODEL_CHECKPOINT:None,
        CHECKPOINT_METADATA_TAG:checkpoint[CHECKPOINT_METADATA_TAG]
    }

def finish_manager_checkpoint(model_checkpoint_path:str, timestamp:str, checkpoint:dict):
    assert checkpoint[CHECKPOINT_STATUS] == CHECKPOINT_STATUS_RUNNING
    return {
            CHECKPOINT_STATUS:CHECKPOINT_STATUS_FINISHED, 
        MODEL_CHECKPOINT:checkpoint[MODEL_CHECKPOINT],
        CHECKPOINT_RUNNING_JOB_ID:checkpoint[CHECKPOINT_RUNNING_JOB_ID], 
        CHECKPOINT_START_TIMESTAMP:checkpoint[CHECKPOINT_START_TIMESTAMP],
        CHECKPOINT_CREATOR_JOB_ID:checkpoint[CHECKPOINT_CREATOR_JOB_ID], 
        CHECKPOINT_CREATE_TIMESTAMP:checkpoint[CHECKPOINT_CREATE_TIMESTAMP],
            CHECKPOINT_STOP_TIMESTAMP:timestamp,
            FINAL_MODEL_CHECKPOINT:model_checkpoint_path,
            CHECKPOINT_METADATA_TAG:checkpoint[CHECKPOINT_METADATA_TAG]
    }

def release_checkpoint_manager(manager_content, job_id, model_checkpoint_path, timestamp):
    new_manager_content = deepcopy(manager_content)
    have_released = False
    for i, checkpoint in enumerate(manager_content[CHECKPOINTS_TAG]):
        if checkpoint[CHECKPOINT_RUNNING_JOB_ID] == job_id and checkpoint[CHECKPOINT_STATUS] == CHECKPOINT_STATUS_RUNNING:
            new_manager_content[CHECKPOINTS_TAG][i] = finish_manager_checkpoint(model_checkpoint_path, timestamp, checkpoint)
            have_released = True
    assert have_released==True
    return new_manager_content 


def __overwrite_manager(new_content, f):
    f.seek(0)
    f.truncate()
    json.dump(new_content, f, indent=4)

def __get_manager_timestamp():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    
def start_job_manager_assesment(job_id:str):
    """Options:
    - returns None to start a new training
    - returns filepath to continue training
    - raises Exception to stop a job in case of no available checkpoint to continue training, 
        before that job checks if all running checkpoints are under a running jobs to resolve deadlocks

    Currently do not take deadlocks into account!!!
    """
    timestamp_now = __get_manager_timestamp()
    with Locker(EXPERIMENT_CHECKPOINT_MANAGER, "r+") as f:
        manager = json.load(f)
        if manager == {}:
            manager[CHECKPOINTS_TAG] = [manager_start_checkpoint(job_id, timestamp_now)]
            __overwrite_manager(manager, f)
            return None
        result = -1
        for i, element in enumerate(manager[CHECKPOINTS_TAG]):
            if element[CHECKPOINT_STATUS] == CHECKPOINT_STATUS_PENDING:
                result = element[MODEL_CHECKPOINT]
                manager[CHECKPOINTS_TAG][i] = run_manager_checkpoint(job_id, timestamp_now, manager[CHECKPOINTS_TAG][i])
                __overwrite_manager(manager, f)
                break
    if result == -1:
        raise Exception("No available trainig to do")
    else:
        return result

def job_out_of_time_checkpoint(job_id, model: Union[torch.nn.Module, FSDP], optimizer, scaler, path: str, rank: int, step: int, batch_size: int, cutoff, joint_loggers: Optional[JointLogger] = None): #TODO params
    """saves the checkpoint"""
    model_path = save_checkpoint(model, optimizer, scaler, path, rank, step, batch_size, cutoff, joint_loggers)
    timestamp_now = __get_manager_timestamp()
    with Locker(EXPERIMENT_CHECKPOINT_MANAGER, "r+") as f:
        manager = json.load(f)
        manager = release_checkpoint_manager(manager, job_id, model_path, timestamp_now)
        manager[CHECKPOINTS_TAG].append(crate_manager_checkpoint(model_path, job_id, timestamp_now))
        __overwrite_manager(manager, f)

def end_training_checkpoint(job_id, model: Union[torch.nn.Module, FSDP], optimizer, scaler, path: str, rank: int, step: int, batch_size: int, cutoff, joint_loggers: Optional[JointLogger] = None):
    """creates last checkpoint and end experiment ending whole experiment"""
    model_path = save_checkpoint(model, optimizer, scaler, path, rank, step, batch_size, cutoff, joint_loggers)
    timestamp_now = __get_manager_timestamp()
    with Locker(EXPERIMENT_CHECKPOINT_MANAGER, "r+") as f:
        manager = json.load(f)
        manager = release_checkpoint_manager(manager, job_id, model_path, timestamp_now)
        __overwrite_manager(manager, f)

def create_slide_checkpoint(job_id, model: Union[torch.nn.Module, FSDP], optimizer, scaler, path: str, rank: int, step: int, batch_size: int, cutoff, joint_loggers: Optional[JointLogger] = None):
    """saves checkpoint and creates a manager checkpoint continuation"""
    model_path = save_checkpoint(model, optimizer, scaler, path, rank, step, batch_size, cutoff, joint_loggers, lr_scheduler_metadata="trapezoidal_slide")
    timestamp_now = __get_manager_timestamp()
    with Locker(EXPERIMENT_CHECKPOINT_MANAGER, "r+") as f:
        manager = json.load(f)
        manager = release_checkpoint_manager(manager, job_id, model_path, timestamp_now)
        manager[CHECKPOINTS_TAG].append(crate_manager_checkpoint(model_path, job_id, timestamp_now, SLIDE_METADATA))
        __overwrite_manager(manager, f)
 

