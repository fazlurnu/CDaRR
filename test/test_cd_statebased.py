import numpy as np
from sim_models.cd_statebased import StateBased
from envs.pairwise_conflict import PairwiseHorConflict

import bluesky as bs

if not getattr(bs, "_joblib_inited", False):
    bs.init(mode="sim", detached=True)
    bs._joblib_inited = True

conf_detection = StateBased()

width = 2
height = 2
horizontal_sep = 50  # in meters
lookahead_time = 15  # seconds
init_speed_ownship = 20  # kts
init_speed_intruder = 20  # kts
dpsi = 180  # degrees
aircraft_type = 'M600'

pairwise = PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
        init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
        init_dpsi=dpsi, aircraft_type_ownship=aircraft_type
    )

states = pairwise._get_states()

# we add one to the lookahead time below so that the drone is iniated in conflict
conf_detection.detect(states, states, horizontal_sep, 100, lookahead_time + 1)

# only show some outputs
nb_to_show = 2
print(conf_detection.rpz[:nb_to_show], conf_detection.confpairs[:nb_to_show],
      conf_detection.tcpa[:nb_to_show], conf_detection.dcpa[:nb_to_show])

pairwise.reset()

print("All imports successful.")