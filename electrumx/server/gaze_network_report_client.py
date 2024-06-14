import json
import requests
import electrumx
from electrumx.lib.hash import hash_to_hex_str
from electrumx.lib.util import class_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from electrumx.server.env import Env
import socket
import urllib3.util.connection as urllib3_cn

# patch requests package to use ipv4 only for client ip address
def allowed_gai_family():
    """
    https://github.com/urllib3/urllib3/blob/main/src/urllib3/util/connection.py
    """
    family = socket.AF_INET
    if urllib3_cn.HAS_IPV6:
        family = socket.AF_INET6 # force ipv6 only if it is available
    return family

urllib3_cn.allowed_gai_family = allowed_gai_family


class GazeNetworkReportClient:
    def __init__(self, env: 'Env'):
        self.logger = class_logger(__name__, self.__class__.__name__)
        self.env = env
    
    def submit_block_report(self, type: str, height: int, block_hash: bytes, event_hash: bytes, cumulative_event_hash: bytes):
        block_hash_str = hash_to_hex_str(block_hash)
        event_hash_str = hash_to_hex_str(event_hash)
        cumulative_event_hash_str = hash_to_hex_str(cumulative_event_hash)
        data = {
            'type': type,
            'clientVersion': electrumx.version,
            'dbVersion': electrumx.gaze_db_version,
            'eventHashVersion': electrumx.gaze_event_hash_version,
            'network': self.env.coin.NET,
            'blockHeight': height,
            'blockHash': block_hash_str,
            'eventHash': event_hash_str,
            'cumulativeEventHash': cumulative_event_hash_str,
        }
        resp = requests.post(f'{self.env.gaze_network_report.url}/v1/report/block', json=data)
        if resp.status_code >= 400:
            self.logger.warning(f'submit_block_report: Failed to submit block report. request_body={json.dumps(data)} status_code={resp.status_code}, text={resp.text}')
        else:
            self.logger.info(f"submit_block_report: Submitted block report to Gaze Network: type {type} height {height}, block_hash {block_hash_str}, event_hash {event_hash_str}, cumulative_event_hash {cumulative_event_hash_str}")
    
    def submit_node_report(self, type: str):
        data = {
            'name': self.env.gaze_network_report.name,
            'type': type,
            'network': self.env.coin.NET,
            'websiteURL': self.env.gaze_network_report.website_url,
            'indexerAPIURL': self.env.gaze_network_report.indexer_api_url,
        }
        resp = requests.post(f'{self.env.gaze_network_report.url}/v1/report/node', json=data)
        if resp.status_code >= 400:
            self.logger.warning(f'submit_node_report: Failed to submit node report. request_body={json.dumps(data)}, status_code={resp.status_code}, text={resp.text}')
        else:
            self.logger.info(f'submit_node_report: Submitted node report to Gaze Network. name={data["name"]}, type={data["type"]}, websiteURL={data["websiteURL"]}, indexerAPIURL={data["indexerAPIURL"]}')
