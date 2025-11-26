from .algorithms.algo_VPG import VPG
from .algorithms.algo_CQL import CQL
from .algorithms.algo_MQR import MQR

def load(config):
    if config['agent'] == 'vpg':
        agent = VPG(config)

    elif config['agent'] == 'CQL':
        agent = CQL(config)

    elif config['agent'] == 'MQR':
        agent = MQR(config)

    return agent