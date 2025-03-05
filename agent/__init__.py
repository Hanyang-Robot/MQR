from .vpg.vpg import VPG
from .vpg.VPG_MQR import MQR
from .vpg.VPG_DR3 import DR3

def load(config):
    if config['agent'] == 'vpg':
        agent = VPG(config)

    # RMO code also includes CQL code.
    elif config['agent'] == 'CQL':
        agent = RMO(config)

    elif config['agent'] == 'CQL+DR3':
        agent = DR3(config)

    elif config['agent'] == 'MQR':
        agent = RMO(config)

    return agent