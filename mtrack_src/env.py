''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('buildpy36')
sys.path.append('Matterport_Simulator/build/')
#import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args

from utils import load_datasets, load_nav_graphs, pad_instr_tokens

# ALFRED imports
from sim_env.thor_env import ThorEnv
from vocab import Vocab
import progressbar
import pprint
import copy
import torch
from collections import defaultdict
from gen.utils.game_util import get_object

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of AiThor environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size=2):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param batch_size:  Used to create the simulator list.
        """
        
        self.sims = []
        self.start_sim(batch_size)
    
    def start_sim(self, batch_size):
        '''
        Start sim
        '''
        for i in range(batch_size):
            env = ThorEnv()
            self.sims.append({"env:": env, "hidx": 0)

class ALFREDBatch():
    ''' Implements the ALFRED navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size=2, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.vocab = {
                'action_low': Vocab(['[PAD]', '[SEG]', '[STOP]']),
                'action_high': Vocab(['[PAD]', '[SEG]', '[STOP]'])
            }
        self.pframe = args.pframe
        self.env = EnvBatch(batch_size=batch_size)

        #TODO Is this used? it is used in initializing the model
        self.feature_size = 2048

        # check if dataset has been preprocessed
        if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
            raise Exception("Dataset not processed; run with --preprocess")

        # load train/valid/tests splits
        with open(args.splits_json) as f:
            splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in splits.items()})

        self.data = []
        if tokenizer:
            self.tok = tokenizer
        i = 0
        if args.preprocess:
            for k, d in splits.items():
                print('Preprocessing {}'.format(k))

                # debugging:
                if args.fast_epoch:
                    d = d[:3]

                for task in progressbar.progressbar(d):
                    # load json file
                    json_path = os.path.join(args.data, k, task['task'], 'traj_data.json')
                    with open(json_path) as f:
                        ex = json.load(f)

                    # copy trajectory
                    r_idx = task['repeat_idx'] # repeat_idx is the index of the annotation for each trajectory
                    traj = ex.copy()

                    # root & split
                    traj['root'] = os.path.join(args.data, task['task'])
                    traj['split'] = k
                    traj['repeat_idx'] = r_idx

                    # BERT pad & tokenize language
                    traj['num'] = {}
                    traj['num']['lang_goal'] = ex['turk_annotations']['anns'][r_idx]['task_desc']
                    traj['num']['lang_instr'] = [x for x in ex['turk_annotations']['anns'][r_idx]['high_descs']] + ['<<stop>>']
                    traj['num']['lang_goal_tok'] = tokenizer.tokenize(ex['turk_annotations']['anns'][r_idx]['task_desc'])
                    traj['num']['lang_instr_tok'] = [tokenizer.tokenize(x) for x in ex['turk_annotations']['anns'][r_idx]['high_descs']]
                    
                    traj['num']['lang_goal_instr_tok_pad'] = []
                    for instr in traj['num']['lang_instr_tok']:
                        padded_instr_tokens, _ = pad_instr_tokens(traj['num']['lang_goal_tok'], instr, args.maxInput)
                        traj['num']['lang_goal_instr_tok_pad'].append(padded_instr_tokens)
                    
                    traj['num']['instr_encodings'] = []
                    for padded_instr_tokens in traj['num']['lang_goal_instr_tok_pad']:
                        traj['num']['instr_encodings'].append(tokenizer.convert_tokens_to_ids(padded_instr_tokens))

                    # Get targets and process actions

                    if "tests" not in k:   # Test has no ground truth target!
                        traj['targets'] = {}
                        targets = defaultdict(list)
                        pddl = ex['plan']['high_pddl']
                        # First get nav target from high_pddl
                        for idx in pddl:
                            action = idx["discrete_action"]["action"]
                            h_idx = idx["high_idx"]
                            # Navgiation actions has only 1 target
                            if action == "GotoLocation":
                                target_name = idx["discrete_action"]["args"][0]
                                target_loc = idx["planner_action"]["location"]
                                targets[h_idx].append({"type": "nav", "target_name": target_name, "target_loc": target_loc})

                        # Then get all interaction targets from low_actions
                        low_actions = ex['plan']['low_actions']
                        for idx in low_actions:
                            action = idx["discrete_action"]["action"]
                            h_idx = idx["high_idx"]

                            # Interaction action might have multiple targets in the same h_idx
                            if "objectId" in idx["api_action"]:
                                target_type = idx["api_action"]["action"]
                                target_name = idx["api_action"]["objectId"]
                                targets[h_idx].append({"type": target_type, "target_name": target_name, "target_loc": None})

                        # Sanity check for # of targets equal to # of instructions
                        #assert len(targets) == len(traj['num']['instr_encodings']), print(json_path)
                        # Due to bug in the dataset, there is an extra h_idx for some slice actions
                        if len(targets) != len(traj['num']['instr_encodings']):
                            wrong_hidx = len(traj['num']['instr_encodings'])
                            action = targets[wrong_hidx]
                            targets[wrong_hidx-1].append(action)
                            del targets[wrong_hidx]
                        
                        # Just one more check
                        assert len(targets) == len(traj['num']['instr_encodings']), print(len(targets), len(traj['num']['instr_encodings']))
                        traj['targets'] = targets

                        # numericalize actions for train/valid splits
                        if 'test' not in k: # expert actions are not available for the test set
                            self.process_actions(ex, traj)

                    # Remove unncessary keys
                    # print(traj)
                    traj.pop('images', None)

                    # check if preprocessing storage folder exists
                    preprocessed_folder = os.path.join(args.data, task['task'], args.pp_folder)
                    if not os.path.isdir(preprocessed_folder):
                        os.makedirs(preprocessed_folder)

                    # save preprocessed json
                    preprocessed_json_path = os.path.join(preprocessed_folder, "ann_%d.json" % r_idx)
                    with open(preprocessed_json_path, 'w') as f:
                        json.dump(traj, f, sort_keys=True, indent=4)

            # save vocab in model path
            vocab_dout_path = os.path.join("snap", args.name, '%s.vocab' % args.pp_folder)
            torch.save(self.vocab, vocab_dout_path)

            # save vocab in data path
            vocab_data_path = os.path.join(args.data, '%s.vocab' % args.pp_folder)
            torch.save(self.vocab, vocab_data_path)

        self.data = splits #TODO This is the replacment for data
        self.seed = seed
        random.seed(self.seed)
        for s in self.data.keys():
            random.shuffle(self.data[s])

        self.ix = 0
        self.batch_size = batch_size

        self.buffered_state_dict = {}

        print(f'ALFREDBatch lazy loaded using splits: {",".join(splits)}')

    def load_task_json(self, args, task):
        '''
        load preprocessed json from disk
        '''
        #TODO Choose 1 from below
        #json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        json_path = os.path.join(args.data, args.split, task['task'], 'traj_data.json')

        with open(json_path) as f:
            data = json.load(f)

        return data
    
    @staticmethod
    def numericalize(vocab, words, train=True):
        '''
        converts words to unique integers
        '''
        return vocab.word2index([w.strip().lower() for w in words], train=train)


    def process_actions(self, ex, traj):
        # deal with missing end high-level action
        self.fix_missing_high_pddl_end_action(ex)

        # end action for low_actions
        end_action = {
            'api_action': {'action': 'NoOp'},
            'discrete_action': {'action': '[STOP]', 'args': {}},
            'high_idx': ex['plan']['high_pddl'][-1]['high_idx']
        }

        # init action_low and action_high
        num_hl_actions = len(ex['plan']['high_pddl'])
        traj['num']['action_low'] = [list() for _ in range(num_hl_actions)]  # temporally aligned with HL actions
        traj['num']['action_high'] = []
        low_to_high_idx = []

        for a in (ex['plan']['low_actions'] + [end_action]):
            # high-level action index (subgoals)
            high_idx = a['high_idx']
            low_to_high_idx.append(high_idx)

            # low-level action (API commands)
            traj['num']['action_low'][high_idx].append({
                'high_idx': a['high_idx'],
                'action': self.vocab['action_low'].word2index(a['discrete_action']['action'], train=True),
                'action_high_args': a['discrete_action']['args'],
            })

            # low-level bounding box (not used in the model)
            if 'bbox' in a['discrete_action']['args']:
                xmin, ymin, xmax, ymax = [float(x) if x != 'NULL' else -1 for x in a['discrete_action']['args']['bbox']]
                traj['num']['action_low'][high_idx][-1]['centroid'] = [
                    (xmin + (xmax - xmin) / 2) / self.pframe,
                    (ymin + (ymax - ymin) / 2) / self.pframe,
                ]
            else:
                traj['num']['action_low'][high_idx][-1]['centroid'] = [-1, -1]

            # low-level interaction mask (Note: this mask needs to be decompressed)
            if 'mask' in a['discrete_action']['args']:
                mask = a['discrete_action']['args']['mask']
            else:
                mask = None
            traj['num']['action_low'][high_idx][-1]['mask'] = mask

            # interaction validity
            def has_interaction(action):
                '''
                check if low-level action is interactive
                '''
                non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '[STOP]', '[PAD]', '[SEG]']
                if any(a in action for a in non_interact_actions):
                    return False
                else:
                    return True

            valid_interact = 1 if has_interaction(a['discrete_action']['action']) else 0
            traj['num']['action_low'][high_idx][-1]['valid_interact'] = valid_interact

        # low to high idx
        traj['num']['low_to_high_idx'] = low_to_high_idx

        # high-level actions
        for a in ex['plan']['high_pddl']:
            traj['num']['action_high'].append({
                'high_idx': a['high_idx'],
                'action': self.vocab['action_high'].word2index(a['discrete_action']['action'], train=True),
                'action_high_args': self.numericalize(self.vocab['action_high'], a['discrete_action']['args']),
            })

        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = len(traj['num']['action_low'])
        lang_instr_seg_len = len(traj['num']['lang_instr'])
        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 0:
            assert (seg_len_diff == 1) # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            self.merge_last_two_low_actions(traj)

            # fix last two for low_to_high_idx and action_high from merge (from: https://github.com/askforalfred/alfred/issues/84)
            traj['num']['low_to_high_idx'][-1] = traj['num']['action_low'][-1][0]["high_idx"]
            traj['num']['low_to_high_idx'][-2] = traj['num']['action_low'][-2][0]["high_idx"]
            traj['num']['action_high'][-1]["high_idx"] = traj['num']['action_high'][-2]["high_idx"]
            traj['num']['action_high'][-2]["high_idx"] = traj['num']['action_high'][-3]["high_idx"]


    def fix_missing_high_pddl_end_action(self, ex):
        '''
        appends a terminal action to a sequence of high-level actions
        '''
        if ex['plan']['high_pddl'][-1]['planner_action']['action'] != 'End':
            ex['plan']['high_pddl'].append({
                'discrete_action': {'action': 'NoOp', 'args': []},
                'planner_action': {'value': 1, 'action': 'End'},
                'high_idx': len(ex['plan']['high_pddl'])
            })


    def merge_last_two_low_actions(self, conv):
        '''
        combines the last two action sequences into one sequence
        '''
        extra_seg = copy.deepcopy(conv['num']['action_low'][-2])
        for sub in extra_seg:
            sub['high_idx'] = conv['num']['action_low'][-3][0]['high_idx']
            conv['num']['action_low'][-3].append(sub)
        del conv['num']['action_low'][-2]
        conv['num']['action_low'][-1][0]['high_idx'] = len(conv['plan']['high_pddl']) - 1


    def size(self):
        return len(self.data)

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size

        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId
    
    def _get_obs(self, init=False):
        obs = []
        for i in range(len(self.batch)):
            task = self.batch[i]
            # Initialize simulator if this is first action
            if init:
                traj_data = model.load_task_json(task)
                r_idx = task['repeat_idx']
                cls.setup_scene(env, traj, r_idx, args, reward_type=self.args.reward_type)
                self.batch[i] = traj_data
            task = self.batch[i]

            # Panorama views
            pan_views = self.create_panorama(self.env.sims[i])
            assert len(pan_views) == 8  # Sanity check, there are 8 views
            views = []
            for i, view in enumerate(pan_views):
                views.append({
                    'viewIndex' : i,
                    'horizon' : view["viewpoint"]["horizon"],
                    'rotation' : view["viewpoint"]["rotation"],
                    'position' : view["viewpoint"]["position"],
                    'image' : view["image"],
                })
            obs.append({
                'instr_encodings' : task['instr_encodings'],
                'teacher' : None,
                'targets' : task['targets']
                'gt_actions': [a['api_action'] for a in task['plan']['low_actions']], # Change this to scene encoding
                'distance': self.get_distance(self.env.sims[i], task['targets'])
                'views': views
            })
            # Get distance reward here??
            
        return obs

    def get_distance(self, env, targets):
        env = env['env']
        hidx = env['hidx']
        curr_target = targets[hidx]

        event = env.last_event
        dist = []
        for target in curr_target:
            if curr_target["type"] == "nav":
                target_pos = tuple([int(i) for i in curr_target["target_loc"].split('|')[1:]])
                curr_pos = event.pose_discrete
                assert(curr_pose[2] in {0, 1, 2, 3})
                assert(target_pos[2] in {0, 1, 2, 3}) # Sanity checks
                dist = np.linalg.norm(np.array([target_pos[0], target_pos[1], target_pos[2], target_pos[3]]) - \
                                    np.array([curr_pos[0], curr_pos[1], curr_pos[2], curr_pos[3]]))
            else:
                curr_pos = event.metadata["agent"]["position"]
                target_obj = get_object(curr_target["target_name"], event.metatdata)
                dist = np.linalg.norm(np.array([curr_pos['x'], curr_pos['y'], curr_pos['z']]) - \
                                    np.array([target_obj['position']['x'], target_obj['position']['y'], target_obj['position']['z']])

        return dist

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        return self._get_obs(init=True)

    #TODO this function forwards environment with predicted action
    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        #self.env.makeActions(actions)
        return self._get_obs()
    

    def create_panorama(self, sim, directions=4, angles=2):
        '''
        Create panorama view for current agent and save it in self.views
        default: 4 NESW views with 2 angles each
        No matter which direction agent is looking, take same image from same view 
        horizion: 30, -30
        rotaion: 0, 90, 180, 270
        Utilize teleport atm TODO: change to natural actions
        '''
        env = sim["env"]
        teleport_actions = [{"action": "Teleport", "horizon": 30}, {"action": "Teleport", "horizon": -30}]
        pan_views = [] # Store all views for current env, so size [# of views]
        degree = (degree + 90) % 360
        init_agent = env.last_event.metatdata["agent"]  # Save initial position info
        for action in teleport_actions:
            env.step(action)
            if event.metadata["lastActionSuccess"]:
                curr_image = Image.fromarray(event.frame)
                camera_horizon = agenv.last_event.metadata["agent"]["cameraHorizon"]
                rotation = env.last_event.metadata["agent"]["rotation"]["y"]
                pos = env.last_event.metadata["agent"]["position"]
                viewpoint_metadata = {"horizon": camera_horizon, "rotation": rotation, "position": pos}
            else:
                '''
                If the action is not possible, append empty image
                '''
                curr_image = Image.fromarray(np.zeros_lie(event.frame))
                camera_horizon = agenv.last_event.metadata["agent"]["cameraHorizon"]
                rotation = degree
                pos = env.last_event.metadata["agent"]["position"]
                viewpoint_metadata = {"horizon": camera_horizon, "rotation": rotation, "position": pos}
            pan_views.append({"image": curr_image, "viewpoint": viewpoint_metadata})

        # Return agent to initial position, also using teleport
        init_action = {
            "action": "TeleportFull", 
            "horizon": init_agent["cameraHorizon"],
            "x": init_agent["position"]["x"],
            "z": init_agent["position"]["z"],
            "y": init_agent["position"]["y"],
            "rotation": init_agent["rotation"]
        }
        env.step(init_action)

        assert env.last_event.metadata["lastActionSuccess"] # Sanity check
        
        return pan_views

    # def create_panorama(env, rotation_steps):
    #     # This is the front view of the agent
    #     initial_agent = env.last_event.metadata["agent"]
    #     curr_image = Image.fromarray(env.last_event.frame)
    #     panorama_frames = [curr_image]
    #     camera_info = [dict(
    #         h_view_angle=env.last_event.metadata["agent"]["rotation"]["y"],
    #         # flip direction of heading angle - negative will be down and positive will be up
    #         v_view_angle=-env.last_event.metadata["agent"]["cameraHorizon"]
    #     )]

    #     # LookUp
    #     env.step({"action": "Teleport", "horizon": 30})
    #     completed_rotations = 0
    #     success = True
    #     angle = env.last_event.metadata["agent"]["rotation"]["y"]

    #     for look in range(rotation_steps):
    #         la_cmd = {'action': 'RotateRight', 'forceAction': True}
    #         event = env.step(la_cmd)

    #         success = success and event.metadata["lastActionSuccess"]
    #         angle = (angle + 90.0) % 360.0
    #         if success:
    #             completed_rotations += 1
    #             curr_image = Image.fromarray(event.frame)
    #             panorama_frames.append(curr_image)
    #             camera_info.append(dict(
    #                 h_view_angle=env.last_event.metadata["agent"]["rotation"]["y"],
    #                 v_view_angle=-env.last_event.metadata["agent"]["cameraHorizon"]
    #             ))
    #         else:
    #             # in this case
    #             panorama_frames.append(Image.fromarray(np.zeros_like(event.frame)))
    #             camera_info.append(dict(
    #                 h_view_angle=angle,
    #                 v_view_angle=-env.last_event.metadata["agent"]["cameraHorizon"]
    #             ))

    #     # at this step we just teleport to the original location
    #     teleport_action = {
    #         'action': 'TeleportFull',
    #         'rotation': initial_agent["rotation"],
    #         'x': initial_agent["position"]['x'],
    #         'z': initial_agent["position"]['z'],
    #         'y': initial_agent["position"]['y'],
    #         'horizon': initial_agent["cameraHorizon"],
    #         'forceAction': True
    #     }
    #     env.step(teleport_action)

    #     assert env.last_event.metadata["lastActionSuccess"], "This shouldn't happen!"

    #     return panorama_frames, camera_info


    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)



    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats
